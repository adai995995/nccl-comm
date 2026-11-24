/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "fabric/device_fabric.h"
#include "bootstrap.h"
#include "channel.h"
#include "comm.h"
#include "net.h"
#include "rma.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin/gin_host_win_stub.h"
#else
#include "gin.h"
#endif
#include "param.h"
#include "proxy.h"
#include "transport.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

NCCL_PARAM(FabricShare, "FABRIC_SHARE", 1);

bool ncclPhysPeerKey::operator==(const ncclPhysPeerKey& o) const {
  return rawHostHash == o.rawHostHash && rawPidHash == o.rawPidHash &&
         memcmp(&gpuUuid, &o.gpuUuid, sizeof(gpuUuid)) == 0;
}

namespace {
struct PhysPeerKeyHash {
  size_t operator()(const ncclPhysPeerKey& k) const {
    uint64_t h[2] = {1, 1};
    eatHash(h, &k.gpuUuid);
    eatHash(h, &k.rawHostHash);
    eatHash(h, &k.rawPidHash);
    return static_cast<size_t>(digestHash(h));
  }
};

struct FabricNvlsCliqueEntry {
  struct ncclNvlsSharedRes* nvlsRes = nullptr;
  int localRanks = 0;
  int nHeads = 0;
  int nChannels = 0;
  int chunkSize = 0;
  int treeMaxChunkSize = 0;
  int fifoLayout = 0;
  std::vector<int> sortedLocalSlots;
  std::vector<int> headSlots;
  struct ncclChannelPeer* ownerPeers[MAXCHANNELS] = {};
  struct ncclDevChannelPeer* ownerDevPeers[MAXCHANNELS] = {};
};

struct PinnedNvlsLease {
  struct ncclNvlsSharedRes* nvlsRes = nullptr;
  int nChannels = 0;
  int localRanks = 0;
  struct ncclChannelPeer* ownerPeers[MAXCHANNELS] = {};
  struct ncclDevChannelPeer* ownerDevPeers[MAXCHANNELS] = {};
};

struct FabricTicketLease {
  bool outerResourcesOwned = false;
  bool proxyPinned = false;
  bool attachSlotHeld = false;
  uint64_t pinnedChanMask = 0;
  int pinnedTpNRanks = 0;
  std::vector<PinnedNvlsLease> pinnedNvls;
};

bool tryIncrementNonZero(int* count) {
  int cur = COMPILER_ATOMIC_LOAD(count, std::memory_order_acquire);
  while (cur > 0) {
    if (COMPILER_ATOMIC_COMPARE_EXCHANGE(count, &cur, cur + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
  }
  return false;
}
} // namespace

struct ncclDeviceFabric {
  int refCount = 1;
  uint64_t fabricId = 0;
  int cudaDev = 0;
  int localTpRank = 0;
  struct ncclSharedResources* sharedRes = nullptr;
  struct ncclMemManager* memManager = nullptr;
  bool cuMemHostEnabled = false;
  bool retired = false;
  std::atomic<bool> tainted{false};

  // Cached plugin handles (copied by value without dereferencing a dead root comm)
  void* netContext = nullptr;
  void* collNetContext = nullptr;
  ncclNet_t* ncclNet = nullptr;
  ncclCollNet_t* ncclCollNet = nullptr;
  int netPluginIndex = -1;
  int ncclNetVer = 0;
  void* rmaContext = nullptr;
  int rmaPluginIndex = -1;
  bool ginSupported = false;
  int ginNumActiveBackends = 0;
  struct ncclGinBackendState ginBackends[NCCL_GIN_MAX_ACTIVE_BACKENDS] = {};
  int maxNvlsChannels = 0;

  std::atomic<bool> attachInFlight{false};
  std::unordered_map<ncclPhysPeerKey, int, PhysPeerKeyHash> peerToSlot;

  std::mutex nvlsMutex;
  std::vector<FabricNvlsCliqueEntry> nvlsCliques;
};

static struct ncclDeviceFabric* retainFabric(struct ncclDeviceFabric* fabric) {
  if (fabric != nullptr) ncclAtomicRefCountIncrement(&fabric->refCount);
  return fabric;
}

static void releaseFabric(struct ncclDeviceFabric* fabric) {
  if (fabric != nullptr && ncclAtomicRefCountDecrement(&fabric->refCount) == 0) {
    delete fabric;
  }
}

static void populateFabricShell(struct ncclComm* shell, const struct ncclDeviceFabric* fabric,
                                struct ncclCudaContext* ctx, int* tpRankSlot, uint32_t* abortSlot) {
  shell->cudaDev = fabric->cudaDev;
  shell->context = ctx;
  shell->rank = 0;
  shell->nRanks = fabric->sharedRes ? fabric->sharedRes->tpNRanks : 1;
  *tpRankSlot = fabric->localTpRank;
  shell->topParentRanks = tpRankSlot;
  *abortSlot = 0;
  shell->abortFlag = abortSlot;
  shell->sharedRes = fabric->sharedRes;
  shell->proxyState = fabric->sharedRes ? fabric->sharedRes->proxyState : nullptr;
  shell->memManager = fabric->memManager;
  shell->netContext = fabric->netContext;
  shell->collNetContext = fabric->collNetContext;
  shell->ncclNet = fabric->ncclNet;
  shell->ncclCollNet = fabric->ncclCollNet;
  shell->netPluginIndex = fabric->netPluginIndex;
  shell->ncclNetVer = fabric->ncclNetVer;
  shell->rmaContext = fabric->rmaContext;
  shell->rmaPluginIndex = fabric->rmaPluginIndex;
  shell->ginState.supported = fabric->ginSupported;
  shell->ginState.numActiveBackends = fabric->ginNumActiveBackends;
  for (int i = 0; i < fabric->ginNumActiveBackends; i++) {
    shell->ginState.backends[i] = fabric->ginBackends[i];
  }
}

static void registerNvlsCliqueOnFabric(struct ncclDeviceFabric* fabric, struct ncclComm* comm,
                                       struct ncclNvlsSharedRes* nvlsRes) {
  if (fabric == nullptr || comm == nullptr || nvlsRes == nullptr || comm->MNNVL) return;

  FabricNvlsCliqueEntry entry{};
  entry.nvlsRes = nvlsRes;
  entry.localRanks = comm->localRanks;
  entry.nHeads = comm->channels[0].nvls.nHeads;
  entry.nChannels = nvlsRes->nChannels;
  entry.chunkSize = nvlsRes->chunkSize;
  entry.treeMaxChunkSize = nvlsRes->treeMaxChunkSize;
  entry.fifoLayout = nvlsRes->fifoLayout;
  entry.sortedLocalSlots.resize(comm->localRanks);
  for (int i = 0; i < comm->localRanks; i++) {
    entry.sortedLocalSlots[i] = comm->topParentRanks[comm->localRankToRank[i]];
  }
  std::sort(entry.sortedLocalSlots.begin(), entry.sortedLocalSlots.end());
  entry.headSlots.resize(entry.nHeads);
  for (int h = 0; h < entry.nHeads; h++) {
    entry.headSlots[h] = comm->topParentRanks[comm->nvlsHeads[h]];
  }
  for (int c = 0; c < entry.nChannels && c < MAXCHANNELS; c++) {
    entry.ownerPeers[c] = comm->channels[c].nvlsPeers;
    entry.ownerDevPeers[c] = comm->channels[c].nvlsDevPeers;
  }

  std::lock_guard<std::mutex> nvlsLock(fabric->nvlsMutex);
  for (const auto& existing : fabric->nvlsCliques) {
    if (existing.nvlsRes == nvlsRes) return;
  }
  fabric->maxNvlsChannels = std::max(fabric->maxNvlsChannels, nvlsRes->nChannels);
  fabric->nvlsCliques.push_back(std::move(entry));
}

int ncclFabricShareEnabled() {
  return ncclParamFabricShare() != 0 ? 1 : 0;
}

uint64_t ncclDeviceFabricTryReserveTicket(struct ncclComm* comm) {
  if (!ncclFabricShareEnabled() || comm == nullptr || comm->context == nullptr) return 0;
  if (comm->sharedRes == nullptr || comm->sharedRes->owner != comm || comm->isGrow) return 0;

  std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
  struct ncclDeviceFabric* fabric = comm->context->fabric;
  if (fabric == nullptr || fabric->retired || fabric->tainted.load(std::memory_order_acquire) ||
      fabric->sharedRes == nullptr) {
    return 0;
  }
  if (fabric->ncclNet != comm->ncclNet) return 0;
  if (fabric->attachInFlight.exchange(true, std::memory_order_acq_rel)) return 0;

  struct ncclProxyState* proxyState = fabric->sharedRes->proxyState;
  bool proxyPinned = false;
  if (proxyState != nullptr) {
    // proxyState->refCount is decremented in commDestroySync BEFORE commFree
    // runs freeChannel or ncclNvlsFree. Pinning proxyState->refCount > 0 here
    // guarantees the owner's channel peers and NVLS resources are still live.
    if (!tryIncrementNonZero(&proxyState->refCount)) {
      fabric->attachInFlight.store(false, std::memory_order_release);
      return 0;
    }
    proxyPinned = true;
  }
  if (!tryIncrementNonZero(&fabric->sharedRes->refCount)) {
    if (proxyPinned) ncclAtomicRefCountDecrement(&proxyState->refCount);
    fabric->attachInFlight.store(false, std::memory_order_release);
    return 0;
  }

  std::unique_ptr<FabricTicketLease> lease(new (std::nothrow) FabricTicketLease{});
  if (!lease) {
    ncclAtomicRefCountDecrement(&fabric->sharedRes->refCount);
    if (proxyPinned) ncclAtomicRefCountDecrement(&proxyState->refCount);
    fabric->attachInFlight.store(false, std::memory_order_release);
    return 0;
  }

  if (fabric->memManager != nullptr) {
    ncclAtomicRefCountIncrement(&fabric->memManager->refCount);
  }
  lease->outerResourcesOwned = true;
  lease->proxyPinned = proxyPinned;
  lease->attachSlotHeld = true;
  lease->pinnedTpNRanks = fabric->sharedRes->tpNRanks;

  // Pin every initialized channel's actual ncclChannelPeer entries so concurrent
  // commFree(A) -> freeChannel() cannot drop peer->refCount to 0 before B runs initChannel().
  for (int c = 0; c < MAXCHANNELS; c++) {
    if (fabric->sharedRes->peers[c] != nullptr) {
      lease->pinnedChanMask |= (1ULL << c);
      for (int r = 0; r < lease->pinnedTpNRanks; r++) {
        ncclAtomicRefCountIncrement(&fabric->sharedRes->peers[c][r].refCount);
      }
    }
  }

  // Also pin any registered NVLS multicast cliques and their nvlsPeers across the window.
  {
    std::lock_guard<std::mutex> nvlsLock(fabric->nvlsMutex);
    for (const auto& entry : fabric->nvlsCliques) {
      if (entry.nvlsRes != nullptr && entry.nvlsRes->inited && tryIncrementNonZero(&entry.nvlsRes->refCount)) {
        PinnedNvlsLease nl{};
        nl.nvlsRes = entry.nvlsRes;
        nl.nChannels = entry.nChannels;
        nl.localRanks = entry.localRanks;
        for (int c = 0; c < entry.nChannels && c < MAXCHANNELS; c++) {
          nl.ownerPeers[c] = entry.ownerPeers[c];
          nl.ownerDevPeers[c] = entry.ownerDevPeers[c];
          if (nl.ownerPeers[c] != nullptr) {
            for (int r = 0; r < entry.localRanks; r++) {
              ncclAtomicRefCountIncrement(&nl.ownerPeers[c][r].refCount);
            }
          }
        }
        lease->pinnedNvls.push_back(nl);
      }
    }
  }

  comm->fabricLease = lease.release();
  comm->reservedFabric = retainFabric(fabric);
  comm->fabricTicketHeld = true;
  return fabric->fabricId;
}

static void releaseLeasePins(struct ncclComm* holderComm, struct ncclDeviceFabric* fabric, FabricTicketLease* lease) {
  struct ncclComm* shell = nullptr;
  if (ncclCalloc(&shell, 1) != ncclSuccess || shell == nullptr) return;
  int tpRank = 0;
  uint32_t zeroAbort = 0;
  populateFabricShell(shell, fabric, holderComm ? holderComm->context : nullptr, &tpRank, &zeroAbort);
  (void)cudaSetDevice(shell->cudaDev);

  // 1. Unpin channel peers (after B has already incremented its own peer->refCount if committed).
  if (lease->pinnedChanMask != 0 && fabric->sharedRes != nullptr) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if ((lease->pinnedChanMask & (1ULL << c)) == 0) continue;
      struct ncclChannelPeer* peers = fabric->sharedRes->peers[c];
      if (peers == nullptr) continue;
      for (int r = 0; r < lease->pinnedTpNRanks; r++) {
        struct ncclChannelPeer* peer = peers + r;
        if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
          for (int b = 0; b < NCCL_MAX_CONNS; b++) {
            if (peer->send[b].transportComm) (void)peer->send[b].transportComm->free(shell, peer->send + b);
            memset(peer->send + b, 0, sizeof(struct ncclConnector));
            if (peer->recv[b].transportComm) (void)peer->recv[b].transportComm->free(shell, peer->recv + b);
            memset(peer->recv + b, 0, sizeof(struct ncclConnector));
          }
        }
      }
    }
    lease->pinnedChanMask = 0;
  }

  // 2. Unpin NVLS cliques and their peer slots.
  for (auto& nl : lease->pinnedNvls) {
    for (int c = 0; c < nl.nChannels && c < MAXCHANNELS; c++) {
      if (nl.ownerPeers[c] == nullptr) continue;
      for (int r = 0; r < nl.localRanks; r++) {
        struct ncclChannelPeer* peer = nl.ownerPeers[c] + r;
        if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
          for (int b = 0; b < NCCL_MAX_CONNS; b++) {
            if (peer->send[b].transportComm) (void)peer->send[b].transportComm->free(shell, peer->send + b);
            memset(peer->send + b, 0, sizeof(struct ncclConnector));
            if (peer->recv[b].transportComm) (void)peer->recv[b].transportComm->free(shell, peer->recv + b);
            memset(peer->recv + b, 0, sizeof(struct ncclConnector));
          }
          if (r == nl.localRanks - 1 && (nl.nvlsRes == nullptr || nl.nvlsRes->nvlsPeersSlab == nullptr)) {
            free(nl.ownerPeers[c]);
            (void)ncclCudaFree(nl.ownerDevPeers[c], shell->memManager);
          }
        }
      }
    }
    shell->nvlsResources = nl.nvlsRes;
    (void)ncclNvlsFree(shell);
    shell->nvlsResources = nullptr;
  }
  lease->pinnedNvls.clear();

  // 3. If outer resources were NOT transferred to comm (rollback path), release them too.
  if (lease->outerResourcesOwned) {
    if (!lease->proxyPinned) shell->proxyState = nullptr;
    if (shell->proxyState != nullptr) {
      (void)ncclProxyStop(shell);
      if (shell->proxyRefCountOld == 0 && shell->proxyState->thread.joinable()) {
        shell->proxyState->thread.join();
        if (shell->proxyState->threadUDS.joinable()) {
          shell->proxyState->threadUDS.join();
        }
      }
    }

    int sharedResRefCount = 1;
    if (shell->sharedRes != nullptr) {
      sharedResRefCount = ncclAtomicRefCountDecrement(&shell->sharedRes->refCount);
      if (sharedResRefCount == 0) {
        (void)ncclGinFinalize(shell);
        for (int c = 0; c < MAXCHANNELS; c++) {
          if (shell->sharedRes->peers[c]) free(shell->sharedRes->peers[c]);
          if (!shell->sharedRes->devPeersSlab && shell->sharedRes->devPeers[c]) {
            (void)ncclCudaFree(shell->sharedRes->devPeers[c], shell->memManager);
          }
        }
        if (shell->sharedRes->devPeersSlab) (void)ncclCudaFree(shell->sharedRes->devPeersSlab, shell->memManager);
        free(shell->sharedRes->tpRankToLocalRank);
        (void)ncclStrongStreamDestruct(&shell->sharedRes->hostStream);
        (void)ncclStrongStreamDestruct(&shell->sharedRes->deviceStream);
        (void)ncclUncapturedStreamPoolDestroy(&shell->sharedRes->uncapturedStreamPool);
        (void)cudaEventDestroy(shell->sharedRes->launchEvent);
        (void)cudaEventDestroy(shell->sharedRes->scratchEvent);
        (void)ncclProxyDestroy(shell);
        ncclDeviceFabricOnSharedResDestroy(shell, shell->sharedRes);
        delete shell->sharedRes;
      }
    }
    if (shell->memManager != nullptr) {
      (void)ncclMemManagerDestroy(shell);
    }
    if (sharedResRefCount == 0) {
      (void)ncclNetFinalize(shell);
      (void)ncclRmaFinalize(shell);
    }
    lease->outerResourcesOwned = false;
  }

  free(shell);
}

void ncclDeviceFabricReleaseTicket(struct ncclComm* comm) {
  if (comm == nullptr) return;
  auto* lease = static_cast<FabricTicketLease*>(comm->fabricLease);
  struct ncclDeviceFabric* fabric = comm->reservedFabric;
  comm->fabricLease = nullptr;
  comm->reservedFabric = nullptr;
  comm->fabricTicketHeld = false;

  if (lease != nullptr) {
    if (fabric != nullptr) {
      if (lease->attachSlotHeld) {
        lease->attachSlotHeld = false;
        fabric->attachInFlight.store(false, std::memory_order_release);
      }
      releaseLeasePins(comm, fabric, lease);
    }
    delete lease;
  }
  releaseFabric(fabric);
}

void ncclDeviceFabricReleaseBound(struct ncclComm* comm) {
  if (comm == nullptr || comm->boundFabric == nullptr) return;
  releaseFabric(comm->boundFabric);
  comm->boundFabric = nullptr;
}

static ncclResult_t destroyTemporaryStandaloneResources(struct ncclComm* comm) {
  struct ncclSharedResources* tmpShared = comm->sharedRes;
  if (tmpShared != nullptr) {
    struct ncclProxyState* tmpProxy = tmpShared->proxyState;
    if (tmpProxy != nullptr) {
      if (tmpProxy->listenSock != nullptr) {
        (void)ncclSocketClose(tmpProxy->listenSock);
        free(tmpProxy->listenSock);
        tmpProxy->listenSock = nullptr;
      }
      (void)ncclIpcSocketClose(&tmpProxy->ipcSock);
      free(tmpProxy->peerAddresses);
      free(tmpProxy->peerAddressesUDS);
      delete tmpProxy;
      tmpShared->proxyState = nullptr;
      comm->proxyState = nullptr;
    }
    NCCLCHECK(ncclGinFinalize(comm));
    free(tmpShared->tpRankToLocalRank);
    NCCLCHECK(ncclStrongStreamDestruct(&tmpShared->hostStream));
    NCCLCHECK(ncclStrongStreamDestruct(&tmpShared->deviceStream));
    NCCLCHECK(ncclUncapturedStreamPoolDestroy(&tmpShared->uncapturedStreamPool));
    CUDACHECK(cudaEventDestroy(tmpShared->launchEvent));
    CUDACHECK(cudaEventDestroy(tmpShared->scratchEvent));
    delete tmpShared;
    comm->sharedRes = nullptr;
  }

  NCCLCHECK(ncclMemManagerDestroy(comm));
  NCCLCHECK(ncclNetFinalize(comm));
  NCCLCHECK(ncclRmaFinalize(comm));
  return ncclSuccess;
}

ncclResult_t ncclDeviceFabricTryBindIndependent(struct ncclComm* comm, bool cuMemHostEnabled) {
  if (!ncclFabricShareEnabled() || comm == nullptr || comm->context == nullptr) return ncclSuccess;
  if (comm->sharedRes == nullptr || comm->sharedRes->owner != comm || comm->isGrow) return ncclSuccess;

  auto* lease = static_cast<FabricTicketLease*>(comm->fabricLease);
  struct ncclDeviceFabric* fabric = comm->reservedFabric;
  if (fabric == nullptr || lease == nullptr || !comm->fabricTicketHeld ||
      fabric->tainted.load(std::memory_order_acquire)) {
    ncclDeviceFabricReleaseTicket(comm);
    return ncclSuccess;
  }

  uint64_t agreedId = comm->peerInfo[0].fabricCandidateId;
  bool unanimous = (agreedId != 0 && agreedId == fabric->fabricId &&
                    !comm->MNNVL && cuMemHostEnabled == fabric->cuMemHostEnabled);
  std::vector<int> slots(comm->nRanks, -1);
  if (unanimous) {
    std::vector<bool> used(fabric->sharedRes->tpNRanks, false);
    for (int r = 0; r < comm->nRanks; r++) {
      if (comm->peerInfo[r].fabricCandidateId != agreedId) {
        unanimous = false;
        break;
      }
      ncclPhysPeerKey key{comm->peerInfo[r].gpuUuid, comm->peerInfo[r].rawHostHash, comm->peerInfo[r].rawPidHash};
      auto it = fabric->peerToSlot.find(key);
      if (it == fabric->peerToSlot.end() || it->second < 0 || it->second >= fabric->sharedRes->tpNRanks ||
          used[it->second]) {
        unanimous = false;
        break;
      }
      used[it->second] = true;
      slots[r] = it->second;
    }
  }

  if (!unanimous) {
    ncclDeviceFabricReleaseTicket(comm);
    return ncclSuccess;
  }

  NCCLCHECK(destroyTemporaryStandaloneResources(comm));

  // Transfer outer sharedRes/proxyState/memManager ownership to comm, while
  // KEEPING lease->pinnedChanMask and lease->pinnedNvls alive on comm->fabricLease
  // until initTransportsRank finishes setupChannel + ncclNvlsSetup!
  comm->sharedRes = fabric->sharedRes;
  comm->proxyState = fabric->sharedRes->proxyState;
  if (comm->proxyState != nullptr && !lease->proxyPinned) {
    ncclAtomicRefCountIncrement(&comm->proxyState->refCount);
  }
  comm->memManager = fabric->memManager;
  lease->outerResourcesOwned = false;
  if (lease->attachSlotHeld) {
    lease->attachSlotHeld = false;
    fabric->attachInFlight.store(false, std::memory_order_release);
  }

  comm->netContext = fabric->netContext;
  comm->collNetContext = fabric->collNetContext;
  comm->ncclNet = fabric->ncclNet;
  comm->ncclCollNet = fabric->ncclCollNet;
  comm->netPluginIndex = fabric->netPluginIndex;
  comm->ncclNetVer = fabric->ncclNetVer;

  comm->rmaContext = fabric->rmaContext;
  comm->rmaPluginIndex = fabric->rmaPluginIndex;

  comm->ginState.supported = fabric->ginSupported;
  comm->ginState.numActiveBackends = fabric->ginNumActiveBackends;
  for (int i = 0; i < fabric->ginNumActiveBackends; i++) {
    comm->ginState.backends[i] = fabric->ginBackends[i];
    comm->ginState.backends[i].ginCommCount = 0;
    memset(comm->ginState.backends[i].ginComms, 0, sizeof(comm->ginState.backends[i].ginComms));
  }

  for (int r = 0; r < comm->nRanks; r++) {
    comm->topParentRanks[r] = slots[r];
  }
  comm->fabricBound = true;
  comm->shareResources = true;
  comm->boundFabric = retainFabric(fabric);

  INFO(NCCL_INIT, "DeviceFabric: bound comm %p (nRanks %d) to fabric 0x%lx",
       comm, comm->nRanks, (unsigned long)fabric->fabricId);
  return ncclSuccess;
}

ncclResult_t ncclDeviceFabricPublishRoot(struct ncclComm* comm) {
  if (!ncclFabricShareEnabled() || comm == nullptr || comm->context == nullptr) return ncclSuccess;
  if (comm->sharedRes == nullptr || comm->sharedRes->owner != comm || comm->MNNVL || comm->isGrow) return ncclSuccess;
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) != 0) return ncclSuccess;

  std::unique_ptr<ncclDeviceFabric> fabric(new (std::nothrow) ncclDeviceFabric{});
  if (!fabric) return ncclSystemError;

  fabric->fabricId = comm->commHash != 0 ? comm->commHash : 1ULL;
  fabric->cudaDev = comm->cudaDev;
  fabric->localTpRank = comm->topParentRanks ? comm->topParentRanks[comm->rank] : comm->rank;
  fabric->sharedRes = comm->sharedRes;
  fabric->memManager = comm->memManager;
  fabric->cuMemHostEnabled = comm->sharedRes->proxyState ? comm->sharedRes->proxyState->cuMemHostEnabled : false;

  fabric->netContext = comm->netContext;
  fabric->collNetContext = comm->collNetContext;
  fabric->ncclNet = comm->ncclNet;
  fabric->ncclCollNet = comm->ncclCollNet;
  fabric->netPluginIndex = comm->netPluginIndex;
  fabric->ncclNetVer = comm->ncclNetVer;
  fabric->rmaContext = comm->rmaContext;
  fabric->rmaPluginIndex = comm->rmaPluginIndex;
  fabric->ginSupported = comm->ginState.supported;
  fabric->ginNumActiveBackends = comm->ginState.numActiveBackends;
  for (int i = 0; i < comm->ginState.numActiveBackends; i++) {
    fabric->ginBackends[i] = comm->ginState.backends[i];
  }
  fabric->maxNvlsChannels = comm->nvlsResources ? comm->nvlsResources->nChannels : 0;

  for (int r = 0; r < comm->nRanks; r++) {
    ncclPhysPeerKey key{comm->peerInfo[r].gpuUuid, comm->peerInfo[r].rawHostHash, comm->peerInfo[r].rawPidHash};
    fabric->peerToSlot.emplace(key, comm->topParentRanks[r]);
  }
  if (comm->nvlsResources != nullptr) {
    registerNvlsCliqueOnFabric(fabric.get(), comm, comm->nvlsResources);
  }

  {
    std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
    if (comm->context->fabric != nullptr) return ncclSuccess;

    struct ncclProxyState* proxyState = comm->sharedRes->proxyState;
    if (proxyState != nullptr) {
      proxyState->abortFlag = &proxyState->serviceAbortFlag;
      if (proxyState->listenSock != nullptr) {
        proxyState->listenSock->abortFlag = &proxyState->serviceAbortFlag;
      }
      proxyState->ipcSock.abortFlag = &proxyState->serviceAbortFlag;
    }

    comm->shareResources = true;
    struct ncclDeviceFabric* published = fabric.release();
    comm->context->fabric = published;
    comm->boundFabric = retainFabric(published);
    INFO(NCCL_INIT, "DeviceFabric: published root fabric 0x%lx peers %d on cudaDev %d",
         (unsigned long)published->fabricId, comm->nRanks, comm->cudaDev);
  }
  return ncclSuccess;
}

bool ncclDeviceFabricNeedsRebuild(const struct ncclComm* comm) {
  if (comm == nullptr || comm->boundFabric == nullptr) return false;
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire) != 0) return false;
  return comm->boundFabric->tainted.load(std::memory_order_acquire);
}

ncclResult_t ncclDeviceFabricRebuildComm(struct ncclComm* comm) {
  if (!ncclDeviceFabricNeedsRebuild(comm)) return ncclSuccess;

  INFO(NCCL_INIT, "DeviceFabric: rebuilding standalone channels for comm %p rank %d after sibling abort",
       comm, comm->rank);
  CUDACHECK(cudaSetDevice(comm->cudaDev));

  if (comm->sharedRes != nullptr) {
    (void)ncclStrongStreamSynchronize(&comm->sharedRes->hostStream);
    (void)ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream);
  }

  bool chanWasInited[MAXCHANNELS] = {};
  int* savedRingUserRanks[MAXCHANNELS] = {};
  int* savedRingRankToIndex[MAXCHANNELS] = {};
  for (int c = 0; c < MAXCHANNELS; c++) {
    if (comm->channels[c].id != -1 && comm->channels[c].peers != nullptr) {
      chanWasInited[c] = true;
      savedRingUserRanks[c] = comm->channels[c].ring.userRanks;
      savedRingRankToIndex[c] = comm->channels[c].ring.rankToIndex;
    }
  }

  if (comm->gproxyConn != nullptr) {
    memset(comm->gproxyConn, 0, sizeof(struct ncclProxyConnector) * comm->nRanks);
  }
  NCCLCHECK(ncclGinHostFinalize(comm));
  NCCLCHECK(ncclRmaProxyFinalize(comm));

  for (int c = 0; c < MAXCHANNELS; c++) {
    NCCLCHECK(freeChannel(comm->channels + c, comm->nRanks, 1, comm->localRanks, comm));
    comm->channels[c].peers = nullptr;
    comm->channels[c].devPeersHostPtr = nullptr;
    comm->channels[c].collnetPeers = nullptr;
    comm->channels[c].collnetDevPeers = nullptr;
    comm->channels[c].nvlsPeers = nullptr;
    comm->channels[c].nvlsDevPeers = nullptr;
    comm->channels[c].id = -1;
  }
  NCCLCHECK(ncclNvlsFree(comm));
  comm->nvlsResources = nullptr;

  // Save old fabric shell before replacing comm's plugin/sharedRes pointers so we
  // can cleanly finalize oldSharedRes if we were its last remaining holder.
  struct ncclComm* oldShell = nullptr;
  NCCLCHECK(ncclCalloc(&oldShell, 1));
  int tpRank = 0;
  uint32_t zeroAbort = 0;
  populateFabricShell(oldShell, comm->boundFabric, comm->context, &tpRank, &zeroAbort);
  oldShell->sharedRes = comm->sharedRes;
  oldShell->proxyState = comm->proxyState;

  if (oldShell->proxyState != nullptr) {
    (void)ncclProxyStop(oldShell);
    if (oldShell->proxyRefCountOld == 0 && oldShell->proxyState->thread.joinable()) {
      oldShell->proxyState->thread.join();
      if (oldShell->proxyState->threadUDS.joinable()) {
        oldShell->proxyState->threadUDS.join();
      }
    }
  }

  int oldSharedRef = 1;
  if (oldShell->sharedRes != nullptr) {
    oldSharedRef = ncclAtomicRefCountDecrement(&oldShell->sharedRes->refCount);
    if (oldSharedRef == 0) {
      (void)ncclGinFinalize(oldShell);
      for (int c = 0; c < MAXCHANNELS; c++) {
        if (oldShell->sharedRes->peers[c]) free(oldShell->sharedRes->peers[c]);
        if (!oldShell->sharedRes->devPeersSlab && oldShell->sharedRes->devPeers[c]) {
          (void)ncclCudaFree(oldShell->sharedRes->devPeers[c], oldShell->memManager);
        }
      }
      if (oldShell->sharedRes->devPeersSlab) (void)ncclCudaFree(oldShell->sharedRes->devPeersSlab, oldShell->memManager);
      free(oldShell->sharedRes->tpRankToLocalRank);
      (void)ncclStrongStreamDestruct(&oldShell->sharedRes->hostStream);
      (void)ncclStrongStreamDestruct(&oldShell->sharedRes->deviceStream);
      (void)ncclUncapturedStreamPoolDestroy(&oldShell->sharedRes->uncapturedStreamPool);
      (void)cudaEventDestroy(oldShell->sharedRes->launchEvent);
      (void)cudaEventDestroy(oldShell->sharedRes->scratchEvent);
      (void)ncclProxyDestroy(oldShell);
      ncclDeviceFabricOnSharedResDestroy(oldShell, oldShell->sharedRes);
      delete oldShell->sharedRes;
    }
  }
  if (oldSharedRef == 0) {
    (void)ncclNetFinalize(oldShell);
    (void)ncclRmaFinalize(oldShell);
  }
  free(oldShell);

  ncclDeviceFabricReleaseBound(comm);
  comm->fabricBound = false;
  comm->shareResources = false;
  comm->sharedRes = nullptr;
  comm->proxyState = nullptr;

  // Re-initialize isolated plugin contexts, sharedRes, and proxyState for comm.
  comm->netContext = nullptr;
  comm->collNetContext = nullptr;
  comm->rmaContext = nullptr;
  comm->rmaPluginIndex = -1;
  comm->ginState.supported = false;
  comm->ginState.numActiveBackends = 0;
  for (int i = 0; i < NCCL_GIN_MAX_ACTIVE_BACKENDS; i++) {
    memset(&comm->ginState.backends[i], 0, sizeof(comm->ginState.backends[i]));
  }

  for (int r = 0; r < comm->nRanks; r++) comm->topParentRanks[r] = r;
  for (int i = 0; i < comm->localRanks; i++) comm->topParentLocalRanks[i] = i;

  struct ncclSharedResources* newShared = nullptr;
  NEW_NOTHROW(newShared, ncclSharedResources);
  if (newShared == nullptr) return ncclSystemError;
  newShared->owner = comm;
  newShared->refCount = 1;
  newShared->uncapturedStreamPool.head = nullptr;
  newShared->tpNRanks = comm->nRanks;
  newShared->tpNLocalRanks = comm->localRanks;
  newShared->magic = comm->magic;
  newShared->tpNChannels = comm->nChannels;
  newShared->tpP2pNChannels = comm->p2pnChannels;
  NCCLCHECK(ncclCalloc(&newShared->tpRankToLocalRank, comm->nRanks));
  memcpy(newShared->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  NCCLCHECK(ncclStrongStreamConstruct(&newShared->deviceStream));
  NCCLCHECK(ncclStrongStreamConstruct(&newShared->hostStream));
  CUDACHECK(cudaEventCreateWithFlags(&newShared->launchEvent, cudaEventDisableTiming));
  CUDACHECK(cudaEventCreateWithFlags(&newShared->scratchEvent, cudaEventDisableTiming));
  comm->sharedRes = newShared;

  NCCLCHECK(ncclNetInit(comm));
  NCCLCHECK(ncclRmaInit(comm));
  NCCLCHECK(ncclGinInit(comm));

  NCCLCHECK(bootstrapSetupStandaloneProxy(comm));
  comm->sharedRes->proxyState->cuMemHostEnabled = (comm->cuMemSupport && ncclCuMemHostEnable());
  NCCLCHECK(ncclProxyCreate(comm));

  // Allocate fresh sharedRes->devPeersSlab and rebind channels with zeroed step counters.
  {
    cudaStream_t deviceStream;
    NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &newShared->deviceStream,
                                      /*concurrent=*/false, &deviceStream));
    size_t totalSharedPeers = (size_t)MAXCHANNELS * comm->nRanks;
    NCCLCHECK(ncclCudaCallocAsync(&newShared->devPeersSlab, totalSharedPeers, deviceStream, comm->memManager,
                                  ncclMemOffload));
    for (int c = 0; c < MAXCHANNELS; c++) {
      newShared->devPeers[c] = newShared->devPeersSlab + (size_t)c * comm->nRanks;
    }
    NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &newShared->deviceStream,
                                      /*concurrent=*/false));
    NCCLCHECK(ncclStrongStreamSynchronize(&newShared->deviceStream));
  }

  for (int c = 0; c < MAXCHANNELS; c++) {
    if (chanWasInited[c] || c < comm->nChannels) {
      NCCLCHECK(initChannel(comm, c));
      if (savedRingUserRanks[c] != nullptr) comm->channels[c].ring.userRanks = savedRingUserRanks[c];
      if (savedRingRankToIndex[c] != nullptr) comm->channels[c].ring.rankToIndex = savedRingRankToIndex[c];
    }
  }

  memset(comm->initAlgoChannels, 0, sizeof(comm->initAlgoChannels));
  memset(comm->connectSend, 0, sizeof(uint64_t) * comm->nRanks);
  memset(comm->connectRecv, 0, sizeof(uint64_t) * comm->nRanks);

  // Reset P2P seen flags and reconnect any already-queued P2P peers for this group.
  bool needP2pConnect = false;
  if (comm->planner.peers != nullptr) {
    for (int peer = 0; peer < comm->nRanks; peer++) {
      comm->planner.peers[peer].sendSeen = false;
      comm->planner.peers[peer].recvSeen = false;
      if (peer == comm->rank) continue;
      if (!ncclIntruQueueEmpty(&comm->planner.peers[peer].sendQueue)) {
        comm->planner.peers[peer].sendSeen = true;
        int round = 0;
        while (peer != comm->p2pSchedule[round].sendRank) round++;
        uint8_t base = ncclP2pChannelBaseForRound(comm, round);
        for (int i = 0; i < comm->p2pnChannelsPerPeer; i++) {
          int ch = ncclP2pChannelForPart(comm->p2pnChannels, base, i);
          if (comm->channels[ch].id == -1) NCCLCHECK(initChannel(comm, ch));
          comm->channels[ch].peers[peer]->send[1].hasSeen = 1;
          comm->channels[ch].peers[peer]->send[1].p2pOnly = 1;
          comm->connectSend[peer] |= (1ULL << ch);
          needP2pConnect = true;
        }
      }
      if (!ncclIntruQueueEmpty(&comm->planner.peers[peer].recvQueue)) {
        comm->planner.peers[peer].recvSeen = true;
        int round = 0;
        while (peer != comm->p2pSchedule[round].recvRank) round++;
        uint8_t base = ncclP2pChannelBaseForRound(comm, round);
        for (int i = 0; i < comm->p2pnChannelsPerPeer; i++) {
          int ch = ncclP2pChannelForPart(comm->p2pnChannels, base, i);
          if (comm->channels[ch].id == -1) NCCLCHECK(initChannel(comm, ch));
          comm->channels[ch].peers[peer]->recv[1].hasSeen = 1;
          comm->channels[ch].peers[peer]->recv[1].p2pOnly = 1;
          comm->connectRecv[peer] |= (1ULL << ch);
          needP2pConnect = true;
        }
      }
    }
  }

  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = {
    &comm->graphs[NCCL_ALGO_TREE], &comm->graphs[NCCL_ALGO_RING],
    &comm->graphs[NCCL_ALGO_COLLNET_DIRECT], &comm->graphs[NCCL_ALGO_COLLNET_CHAIN],
    &comm->graphs[NCCL_ALGO_NVLS], &comm->graphs[NCCL_ALGO_NVLS], &comm->graphs[NCCL_ALGO_TREE]
  };
  if (comm->runtimeConn) {
    NCCLCHECK(ncclNvlsSetup(comm, nullptr));
    if (comm->config.collnetEnable) (void)ncclCollNetSetup(comm, nullptr, graphs);
  } else {
    NCCLCHECK(ncclTransportRingConnect(comm));
    NCCLCHECK(ncclTransportTreeConnect(comm));
    NCCLCHECK(ncclNvlsSetup(comm, nullptr));
    NCCLCHECK(ncclNvlsBufferSetup(comm));
    NCCLCHECK(ncclTransportPatConnect(comm));
    NCCLCHECK(ncclNvlsTreeConnect(comm));
    if (comm->config.collnetEnable) {
      (void)ncclCollNetSetup(comm, nullptr, graphs);
      NCCLCHECK(ncclCollNetChainBufferSetup(comm));
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY + 1) {
        NCCLCHECK(ncclCollNetDirectBufferSetup(comm));
      }
    }
    struct ncclProxyConnector proxyConn;
    NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn));
    NCCLCHECK(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int),
                                    nullptr, 0));
  }
  if (needP2pConnect) {
    NCCLCHECK(ncclTransportP2pSetup(comm, nullptr, 1));
  }

  NCCLCHECK(ncclStrongStreamSynchronize(&newShared->hostStream));
  NCCLCHECK(ncclStrongStreamSynchronize(&newShared->deviceStream));
  NCCLCHECK(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
                                      comm->localRankToRank[0]));
  return ncclSuccess;
}

bool ncclDeviceFabricMatchNvlsClique(struct ncclComm* comm,
                                     struct ncclNvlsSharedRes** outRes,
                                     struct ncclChannelPeer* outOwnerPeers[64],
                                     struct ncclDevChannelPeer* outOwnerDevPeers[64],
                                     std::vector<int>* outSlotMap) {
  *outRes = nullptr;
  if (!ncclFabricShareEnabled() || comm == nullptr || comm->context == nullptr || comm->MNNVL) return false;

  struct ncclDeviceFabric* fabric = comm->boundFabric ? retainFabric(comm->boundFabric) : nullptr;
  if (fabric == nullptr) {
    std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
    if (comm->context->fabric != nullptr && comm->context->fabric->sharedRes == comm->sharedRes) {
      fabric = retainFabric(comm->context->fabric);
    }
  }
  if (fabric == nullptr || fabric->sharedRes != comm->sharedRes ||
      fabric->tainted.load(std::memory_order_acquire)) {
    releaseFabric(fabric);
    return false;
  }

  std::vector<int> sortedLocal(comm->localRanks);
  for (int i = 0; i < comm->localRanks; i++) {
    sortedLocal[i] = comm->topParentRanks[comm->localRankToRank[i]];
  }
  std::sort(sortedLocal.begin(), sortedLocal.end());

  int nHeads = comm->channels[0].nvls.nHeads;
  bool matched = false;
  {
    std::lock_guard<std::mutex> nvlsLock(fabric->nvlsMutex);
    for (const auto& entry : fabric->nvlsCliques) {
      if (entry.nvlsRes == nullptr || !entry.nvlsRes->inited) continue;
      if (entry.localRanks != comm->localRanks || entry.nHeads != nHeads) continue;
      if (entry.sortedLocalSlots != sortedLocal) continue;

      outSlotMap->assign(comm->localRanks, 0);
      bool headsMatch = true;
      for (int h = 0; h < nHeads; h++) {
        int headSlot = comm->topParentRanks[comm->nvlsHeads[h]];
        auto it = std::find(entry.headSlots.begin(), entry.headSlots.end(), headSlot);
        if (it == entry.headSlots.end()) {
          headsMatch = false;
          break;
        }
        (*outSlotMap)[h] = static_cast<int>(it - entry.headSlots.begin());
      }
      if (!headsMatch) continue;
      for (int r = nHeads; r < comm->localRanks; r++) (*outSlotMap)[r] = r;

      *outRes = entry.nvlsRes;
      for (int c = 0; c < entry.nChannels && c < 64; c++) {
        outOwnerPeers[c] = entry.ownerPeers[c];
        outOwnerDevPeers[c] = entry.ownerDevPeers[c];
      }
      matched = true;
      break;
    }
  }
  releaseFabric(fabric);
  return matched;
}

void ncclDeviceFabricRegisterNvlsClique(struct ncclComm* comm, struct ncclNvlsSharedRes* nvlsRes) {
  if (!ncclFabricShareEnabled() || comm == nullptr || nvlsRes == nullptr || comm->context == nullptr || comm->MNNVL)
    return;

  struct ncclDeviceFabric* fabric = comm->boundFabric ? retainFabric(comm->boundFabric) : nullptr;
  if (fabric == nullptr) {
    std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
    if (comm->context->fabric != nullptr && comm->context->fabric->sharedRes == comm->sharedRes) {
      fabric = retainFabric(comm->context->fabric);
    }
  }
  if (fabric == nullptr) return;
  registerNvlsCliqueOnFabric(fabric, comm, nvlsRes);
  releaseFabric(fabric);
}

void ncclDeviceFabricUnregisterNvlsClique(struct ncclComm* comm, struct ncclNvlsSharedRes* nvlsRes) {
  if (comm == nullptr || nvlsRes == nullptr || comm->context == nullptr) return;
  struct ncclDeviceFabric* fabric = comm->boundFabric ? retainFabric(comm->boundFabric) : nullptr;
  if (fabric == nullptr) {
    std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
    if (comm->context->fabric != nullptr) {
      fabric = retainFabric(comm->context->fabric);
    }
  }
  if (fabric == nullptr) return;

  {
    std::lock_guard<std::mutex> nvlsLock(fabric->nvlsMutex);
    fabric->nvlsCliques.erase(
      std::remove_if(fabric->nvlsCliques.begin(), fabric->nvlsCliques.end(),
                     [nvlsRes](const FabricNvlsCliqueEntry& e) { return e.nvlsRes == nvlsRes; }),
      fabric->nvlsCliques.end());
  }
  releaseFabric(fabric);
}

int ncclDeviceFabricMaxNvlsChannels(struct ncclComm* comm) {
  if (comm == nullptr || comm->context == nullptr) return 0;
  if (comm->boundFabric != nullptr && comm->boundFabric->sharedRes == comm->sharedRes) {
    return comm->boundFabric->maxNvlsChannels;
  }
  std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
  if (comm->context->fabric == nullptr || comm->context->fabric->sharedRes != comm->sharedRes) return 0;
  return comm->context->fabric->maxNvlsChannels;
}

void ncclDeviceFabricRetireOnAbort(struct ncclComm* comm) {
  if (comm == nullptr) return;
  if (comm->boundFabric != nullptr) {
    comm->boundFabric->retired = true;
    comm->boundFabric->tainted.store(true, std::memory_order_release);
  }
  if (comm->context != nullptr && comm->sharedRes != nullptr) {
    struct ncclDeviceFabric* retired = nullptr;
    {
      std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
      if (comm->context->fabric != nullptr && comm->context->fabric->sharedRes == comm->sharedRes) {
        retired = comm->context->fabric;
        retired->retired = true;
        retired->tainted.store(true, std::memory_order_release);
        comm->context->fabric = nullptr;
      }
    }
    releaseFabric(retired);
  }
}

void ncclDeviceFabricOnSharedResDestroy(struct ncclComm* comm, struct ncclSharedResources* sharedRes) {
  if (comm == nullptr || comm->context == nullptr || sharedRes == nullptr) return;
  struct ncclDeviceFabric* dying = nullptr;
  {
    std::lock_guard<std::mutex> lock(comm->context->fabricMutex);
    if (comm->context->fabric != nullptr && comm->context->fabric->sharedRes == sharedRes) {
      dying = comm->context->fabric;
      dying->retired = true;
      comm->context->fabric = nullptr;
    }
  }
  releaseFabric(dying);
}
