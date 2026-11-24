/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "graph/slot_route_memo.h"
#include "graph/topo.h"
#include "graph/topo_cache.h"
#include "comm.h"
#include "transport.h"
#include "param.h"
#include <algorithm>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

NCCL_PARAM(TopoRouteMemo, "TOPO_ROUTE_MEMO", 1);
NCCL_PARAM(TopoRouteMemoMaxEntries, "TOPO_ROUTE_MEMO_MAX_ENTRIES", 64);

namespace {
struct SlotMemoKey {
  uint64_t nodeDigest[2];
  int nLocalRanks;
  uint8_t entryKind;
  bool hasRemotePeers;
  bool cuMemGdrSupport;
  bool collnetEnable;
  bool nvlsSupport;
  bool allSameShmDev;
  bool anySamePid;
  int minDriverVersion;
  int minCompCap;
  int maxCompCap;
  int p2pLevel;
  int pxnDisable;
  uint64_t envHash;

  bool operator==(const SlotMemoKey& o) const {
    return nodeDigest[0] == o.nodeDigest[0] && nodeDigest[1] == o.nodeDigest[1] &&
           nLocalRanks == o.nLocalRanks && entryKind == o.entryKind &&
           hasRemotePeers == o.hasRemotePeers &&
           cuMemGdrSupport == o.cuMemGdrSupport && collnetEnable == o.collnetEnable &&
           nvlsSupport == o.nvlsSupport && allSameShmDev == o.allSameShmDev &&
           anySamePid == o.anySamePid && minDriverVersion == o.minDriverVersion &&
           minCompCap == o.minCompCap && maxCompCap == o.maxCompCap &&
           p2pLevel == o.p2pLevel && pxnDisable == o.pxnDisable && envHash == o.envHash;
  }
};

struct SlotMemoKeyHash {
  size_t operator()(const SlotMemoKey& k) const {
    uint64_t h[2] = {1, 1};
    eatHash(h, &k, sizeof(k));
    return static_cast<size_t>(digestHash(h));
  }
};

uint64_t computeGraphEnvHash() {
  static const char* const kEnvVars[] = {
    "NCCL_NVLS_ENABLE", "NCCL_NVLS_NCHANNELS", "NCCL_COLLNET_ENABLE",
    "NCCL_CROSS_NIC", "NCCL_ALGO", "NCCL_PROTO", "NCCL_NTHREADS",
    "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS", "NCCL_MIN_NRINGS", "NCCL_MAX_NRINGS",
    "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_IGNORE_DISABLED_P2P",
    "NCCL_SHM_DISABLE", "NCCL_NET_GDR_LEVEL", "NCCL_NET_GDR_READ", "NCCL_NET_GDR_C2C",
    "NCCL_NET_DISABLE_INTRA", "NCCL_PXN_DISABLE", "NCCL_PXN_C2C", "NCCL_P2P_PXN_LEVEL",
    "NCCL_NVB_DISABLE", "NCCL_NET_FORCE_MERGE", "NCCL_NET_MERGE_LEVEL",
    "NCCL_NET_MERGE_POLICY", "NCCL_NETDEVS_POLICY", "NCCL_GRAPH_FILE"
  };
  uint64_t h[2] = {1, 1};
  for (const char* name : kEnvVars) {
    const char* val = ncclGetEnv(name);
    eatHash(h, name, strlen(name));
    if (val != nullptr) eatHash(h, val, strlen(val));
  }
  return digestHash(h);
}

ncclResult_t cloneTopoSystem(const struct ncclTopoSystem* src, const int* rankMap, int rankMapLen,
                             uint64_t overrideHostHash, struct ncclTopoSystem** out) {
  *out = nullptr;
  struct ncclTopoSystem* dst = nullptr;
  NCCLCHECK(ncclCalloc(&dst, 1));
  memcpy(dst, src, sizeof(*dst));
  dst->selectNetsCache = nullptr;
  if (overrideHostHash != 0 && dst->nHosts > 0) {
    dst->hostHashes[dst->systemId] = overrideHostHash;
  }

  // Clear path pointers first so partial failure can safely call ncclTopoFree(dst).
  for (int t1 = 0; t1 < NCCL_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < dst->nodes[t1].count; n++) {
      for (int t2 = 0; t2 < NCCL_TOPO_NODE_TYPES; t2++) {
        dst->nodes[t1].nodes[n].paths[t2] = nullptr;
      }
    }
  }

  auto relocate = [src, dst](const void* ptr) -> void* {
    if (ptr == nullptr) return nullptr;
    ptrdiff_t off = reinterpret_cast<const char*>(ptr) - reinterpret_cast<const char*>(src);
    return reinterpret_cast<char*>(dst) + off;
  };

  for (int t1 = 0; t1 < NCCL_TOPO_NODE_TYPES; t1++) {
    for (int n = 0; n < dst->nodes[t1].count; n++) {
      struct ncclTopoNode* dnode = &dst->nodes[t1].nodes[n];
      const struct ncclTopoNode* snode = &src->nodes[t1].nodes[n];
      dnode->used = 0;
      if (t1 == GPU) {
        dnode->gpu.parent = static_cast<struct ncclTopoNode*>(relocate(snode->gpu.parent));
        int r = snode->gpu.rank;
        if (rankMap != nullptr) {
          if (r < 0 || r >= rankMapLen || rankMap[r] < 0) {
            ncclTopoFree(dst);
            return ncclInternalError;
          }
          dnode->gpu.rank = rankMap[r];
        }
      }
      for (int l = 0; l < dnode->nlinks; l++) {
        dnode->links[l].remNode = static_cast<struct ncclTopoNode*>(relocate(snode->links[l].remNode));
      }
      for (int t2 = 0; t2 < NCCL_TOPO_NODE_TYPES; t2++) {
        if (snode->paths[t2] == nullptr) continue;
        int count2 = src->nodes[t2].count;
        if (count2 <= 0) continue;
        if (ncclCalloc(&dnode->paths[t2], count2) != ncclSuccess) {
          ncclTopoFree(dst);
          return ncclSystemError;
        }
        for (int p = 0; p < count2; p++) {
          dnode->paths[t2][p] = snode->paths[t2][p];
          dnode->paths[t2][p].list = nullptr;
          if (snode->paths[t2][p].list != nullptr) {
            int cap = std::max(1, std::max(snode->paths[t2][p].capacity, snode->paths[t2][p].count));
            if (ncclCalloc(&dnode->paths[t2][p].list, cap) != ncclSuccess) {
              ncclTopoFree(dst);
              return ncclSystemError;
            }
            dnode->paths[t2][p].capacity = cap;
            for (int h = 0; h < snode->paths[t2][p].count; h++) {
              dnode->paths[t2][p].list[h] =
                static_cast<struct ncclTopoLink*>(relocate(snode->paths[t2][p].list[h]));
            }
          }
        }
      }
    }
  }

  *out = dst;
  return ncclSuccess;
}

struct CachedSlotEntry {
  SlotMemoKey key;
  struct ncclTopoSystem* canonicalSystem = nullptr;
  bool graphsReady = false;
  struct ncclTopoGraph graphs[5] = {};

  ~CachedSlotEntry() {
    if (canonicalSystem != nullptr) ncclTopoFree(canonicalSystem);
  }
};

struct MemoCache {
  std::mutex mutex;
  std::list<std::shared_ptr<CachedSlotEntry>> lru;
  std::unordered_map<SlotMemoKey, std::list<std::shared_ptr<CachedSlotEntry>>::iterator, SlotMemoKeyHash> index;
};

MemoCache& memoCache() {
  static MemoCache instance;
  return instance;
}

SlotMemoKey makeKey(struct ncclComm* comm, const uint64_t nodeDigest[2], const std::vector<int>& localRanks,
                    bool includeGraphFlags, bool collnetEnable, bool nvlsSupport) {
  SlotMemoKey key;
  memset(&key, 0, sizeof(key));
  key.nodeDigest[0] = nodeDigest[0];
  key.nodeDigest[1] = nodeDigest[1];
  key.nLocalRanks = static_cast<int>(localRanks.size());
  key.entryKind = includeGraphFlags ? 1 : 0;
  key.hasRemotePeers = comm->nRanks > key.nLocalRanks;
  key.cuMemGdrSupport = comm->cuMemGdrSupport;
  key.collnetEnable = includeGraphFlags ? collnetEnable : false;
  key.nvlsSupport = includeGraphFlags ? nvlsSupport : false;
  key.allSameShmDev = true;
  key.anySamePid = false;
  if (!localRanks.empty()) {
    uint64_t shm0 = comm->peerInfo[localRanks[0]].shmDev;
    for (size_t i = 0; i < localRanks.size(); i++) {
      const auto& pi = comm->peerInfo[localRanks[i]];
      if (pi.shmDev != shm0) key.allSameShmDev = false;
      for (size_t j = i + 1; j < localRanks.size(); j++) {
        if (pi.pidHash == comm->peerInfo[localRanks[j]].pidHash) key.anySamePid = true;
      }
    }
  }
  key.minDriverVersion = comm->minDriverVersion;
  key.minCompCap = comm->minCompCap;
  key.maxCompCap = comm->maxCompCap;
  int p2pLevel = -1;
  (void)ncclGetUserP2pLevel(&p2pLevel);
  key.p2pLevel = p2pLevel;
  key.pxnDisable = ncclPxnDisable(comm);
  key.envHash = computeGraphEnvHash();
  return key;
}
} // namespace

int ncclSlotRouteMemoEnabled() {
  return (ncclParamTopoRouteMemo() != 0 && ncclTopoCacheMode() >= 2) ? 1 : 0;
}

void ncclSlotRouteMemoClear() {
  MemoCache& mc = memoCache();
  std::lock_guard<std::mutex> lock(mc.mutex);
  mc.index.clear();
  mc.lru.clear();
}

ncclResult_t ncclTopoComputeNormalizedNodeDigest(struct ncclXml* xml, const std::vector<int>& localRanks,
                                                 uint64_t outDigest[2]) {
  outDigest[0] = outDigest[1] = 0;
  if (xml == nullptr || localRanks.empty()) return ncclInternalError;

  // Save and normalize rank -> localSlot and cpu.host_hash -> 0 in-place, pack, then restore.
  struct SavedAttr {
    ncclXmlNode* node;
    const char* key;
    std::string oldVal;
  };
  std::vector<SavedAttr> saved;
  std::unordered_map<uint64_t, uint64_t> guidToCanon;
  uint64_t nextCanonGuid = 1;
  std::vector<int> rankToSlot;
  int maxRank = -1;
  for (int r : localRanks) maxRank = std::max(maxRank, r);
  rankToSlot.assign(maxRank + 1, -1);
  for (int s = 0; s < int(localRanks.size()); s++) rankToSlot[localRanks[s]] = s;

  for (int i = 0; i < xml->maxIndex; i++) {
    ncclXmlNode* node = &xml->nodes[i];
    if (strcmp(node->name, "gpu") == 0) {
      const char* val = nullptr;
      NCCLCHECK(xmlGetAttr(node, "rank", &val));
      if (val != nullptr) {
        int r = strtol(val, nullptr, 0);
        if (r >= 0 && r < int(rankToSlot.size()) && rankToSlot[r] >= 0) {
          saved.push_back({node, "rank", val});
          NCCLCHECK(xmlSetAttrInt(node, "rank", rankToSlot[r]));
        }
      }
    } else if (strcmp(node->name, "cpu") == 0) {
      const char* val = nullptr;
      NCCLCHECK(xmlGetAttr(node, "host_hash", &val));
      if (val != nullptr) {
        saved.push_back({node, "host_hash", val});
        NCCLCHECK(xmlSetAttrLong(node, "host_hash", 0));
      }
    } else if (strcmp(node->name, "net") == 0 || strcmp(node->name, "gin") == 0 ||
               strcmp(node->name, "rma") == 0) {
      const char* val = nullptr;
      NCCLCHECK(xmlGetAttr(node, "guid", &val));
      if (val != nullptr) {
        uint64_t rawGuid = strtoull(val, nullptr, 16);
        uint64_t canonGuid = 0;
        if (rawGuid != 0) {
          auto it = guidToCanon.find(rawGuid);
          if (it == guidToCanon.end()) {
            canonGuid = nextCanonGuid++;
            guidToCanon.emplace(rawGuid, canonGuid);
          } else {
            canonGuid = it->second;
          }
        }
        saved.push_back({node, "guid", val});
        NCCLCHECK(xmlSetAttrUint64(node, "guid", canonGuid));
      }
    }
  }

  std::string packed;
  ncclResult_t ret = ncclTopoPackXml(xml, &packed);
  for (const auto& item : saved) {
    (void)xmlSetAttr(item.node, item.key, item.oldVal.c_str());
  }
  NCCLCHECK(ret);

  uint64_t a[2] = {1, 1}, b[2] = {0x736c6f746d656d6fULL, 0x6e63636c67727068ULL};
  eatHash(a, packed.data(), packed.size());
  eatHash(b, packed.data(), packed.size());
  outDigest[0] = digestHash(a);
  outDigest[1] = digestHash(b);
  return ncclSuccess;
}

ncclResult_t ncclSlotRouteMemoTryGetSystem(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                           const std::vector<int>& localRanks, bool* hit) {
  *hit = false;
  if (!ncclSlotRouteMemoEnabled() || comm->MNNVL || (nodeDigest[0] == 0 && nodeDigest[1] == 0)) return ncclSuccess;

  SlotMemoKey sysKey = makeKey(comm, nodeDigest, localRanks, /*includeGraphFlags=*/false, false, false);
  if (sysKey.anySamePid || !sysKey.allSameShmDev) return ncclSuccess;

  std::shared_ptr<CachedSlotEntry> found;
  {
    MemoCache& mc = memoCache();
    std::lock_guard<std::mutex> lock(mc.mutex);
    auto it = mc.index.find(sysKey);
    if (it != mc.index.end()) {
      mc.lru.splice(mc.lru.begin(), mc.lru, it->second);
      found = *it->second;
    }
  }
  if (!found || found->canonicalSystem == nullptr) return ncclSuccess;

  NCCLCHECK(cloneTopoSystem(found->canonicalSystem, localRanks.data(), static_cast<int>(localRanks.size()),
                            getHostHash(), &comm->topo));
  *hit = true;
  INFO(NCCL_GRAPH, "TOPO/MEMO system hit localRanks %d", int(localRanks.size()));
  return ncclSuccess;
}

ncclResult_t ncclSlotRouteMemoTryGetGraphs(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                           const std::vector<int>& localRanks,
                                           struct ncclTopoGraph* graphs[5], bool* hit) {
  *hit = false;
  if (!ncclSlotRouteMemoEnabled() || comm->MNNVL || (nodeDigest[0] == 0 && nodeDigest[1] == 0)) return ncclSuccess;

  SlotMemoKey graphKey = makeKey(comm, nodeDigest, localRanks, /*includeGraphFlags=*/true,
                                 comm->config.collnetEnable != 0, comm->nvlsSupport != 0);
  if (graphKey.anySamePid || !graphKey.allSameShmDev) return ncclSuccess;

  std::shared_ptr<CachedSlotEntry> found;
  {
    MemoCache& mc = memoCache();
    std::lock_guard<std::mutex> lock(mc.mutex);
    auto it = mc.index.find(graphKey);
    if (it != mc.index.end() && (*it->second)->graphsReady) {
      mc.lru.splice(mc.lru.begin(), mc.lru, it->second);
      found = *it->second;
    }
  }
  if (!found) return ncclSuccess;

  int nLocal = static_cast<int>(localRanks.size());
  for (int g = 0; g < 5; g++) {
    *graphs[g] = found->graphs[g];
    int count = graphs[g]->nChannels * nLocal;
    for (int i = 0; i < count && i < MAXCHANNELS * NCCL_TOPO_MAX_NODES; i++) {
      int slot = graphs[g]->intra[i];
      if (slot >= 0 && slot < nLocal) graphs[g]->intra[i] = localRanks[slot];
    }
  }
  *hit = true;
  INFO(NCCL_GRAPH, "TOPO/MEMO graphs hit localRanks %d nChannels %d", nLocal, graphs[0]->nChannels);
  return ncclSuccess;
}

ncclResult_t ncclSlotRouteMemoStore(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                    const std::vector<int>& localRanks,
                                    struct ncclTopoGraph* const graphs[5]) {
  if (!ncclSlotRouteMemoEnabled() || comm->MNNVL || comm->topo == nullptr) return ncclSuccess;
  if (nodeDigest[0] == 0 && nodeDigest[1] == 0) return ncclSuccess;
  int nLocal = static_cast<int>(localRanks.size());
  if (nLocal <= 0 || comm->topo->nodes[GPU].count != nLocal) return ncclSuccess;

  SlotMemoKey sysKey = makeKey(comm, nodeDigest, localRanks, /*includeGraphFlags=*/false, false, false);
  SlotMemoKey graphKey = makeKey(comm, nodeDigest, localRanks, /*includeGraphFlags=*/true,
                                 comm->config.collnetEnable != 0, comm->nvlsSupport != 0);
  if (sysKey.anySamePid || !sysKey.allSameShmDev) return ncclSuccess;

  int maxRank = -1;
  for (int r : localRanks) maxRank = std::max(maxRank, r);
  std::vector<int> rankToSlot(maxRank + 1, -1);
  for (int s = 0; s < nLocal; s++) rankToSlot[localRanks[s]] = s;

  int64_t limit = ncclParamTopoRouteMemoMaxEntries();
  if (limit <= 0) return ncclSuccess;

  auto insertEntry = [&](const SlotMemoKey& key, bool storeGraphs) -> ncclResult_t {
    auto entry = std::make_shared<CachedSlotEntry>();
    entry->key = key;
    NCCLCHECK(cloneTopoSystem(comm->topo, rankToSlot.data(), static_cast<int>(rankToSlot.size()),
                              /*overrideHostHash=*/0, &entry->canonicalSystem));
    if (storeGraphs) {
      entry->graphsReady = true;
      for (int g = 0; g < 5; g++) {
        entry->graphs[g] = *graphs[g];
        int count = entry->graphs[g].nChannels * nLocal;
        for (int i = 0; i < count && i < MAXCHANNELS * NCCL_TOPO_MAX_NODES; i++) {
          int r = entry->graphs[g].intra[i];
          if (r >= 0 && r < int(rankToSlot.size()) && rankToSlot[r] >= 0) {
            entry->graphs[g].intra[i] = rankToSlot[r];
          }
        }
      }
    }
    MemoCache& mc = memoCache();
    std::lock_guard<std::mutex> lock(mc.mutex);
    auto it = mc.index.find(key);
    if (it != mc.index.end()) {
      if (storeGraphs && !(*it->second)->graphsReady) {
        for (int g = 0; g < 5; g++) (*it->second)->graphs[g] = entry->graphs[g];
        (*it->second)->graphsReady = true;
      }
      return ncclSuccess;
    }
    while (int(mc.lru.size()) >= limit) {
      mc.index.erase(mc.lru.back()->key);
      mc.lru.pop_back();
    }
    mc.lru.push_front(entry);
    mc.index.emplace(key, mc.lru.begin());
    return ncclSuccess;
  };

  NCCLCHECK(insertEntry(sysKey, /*storeGraphs=*/false));
  NCCLCHECK(insertEntry(graphKey, /*storeGraphs=*/true));
  return ncclSuccess;
}
