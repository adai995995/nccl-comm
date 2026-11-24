/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "fabric/channel_slab.h"
#include "comm.h"
#include "alloc.h"
#include <cstdlib>
#include <cstring>

namespace {
constexpr size_t kSlabAlign = 256;

inline size_t alignUp(size_t v, size_t a = kSlabAlign) {
  return (v + a - 1) & ~(a - 1);
}
} // namespace

ncclResult_t ncclChannelSlabAlloc(struct ncclComm* comm) {
  static_assert(MAXCHANNELS <= 64, "ncclCommDevMetaSlab assumes MAXCHANNELS <= 64");
  if (comm->devMetaSlab != nullptr) return ncclSuccess;

  ncclResult_t ret = ncclSuccess;
  cudaStream_t deviceStream;
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));

  // Batch sharedRes->devPeers[0..MAXCHANNELS-1] into a single device allocation
  // instead of MAXCHANNELS separate 2 MiB VMM allocations.
  if (comm->sharedRes->devPeers[0] == nullptr) {
    size_t totalSharedPeers = (size_t)MAXCHANNELS * (size_t)comm->sharedRes->tpNRanks;
    NCCLCHECKGOTO(ncclCudaCallocAsync(&comm->sharedRes->devPeersSlab, totalSharedPeers, deviceStream,
                                      comm->memManager, ncclMemOffload),
                  ret, exit);
    for (int c = 0; c < MAXCHANNELS; c++) {
      comm->sharedRes->devPeers[c] = comm->sharedRes->devPeersSlab + (size_t)c * comm->sharedRes->tpNRanks;
    }
  }

  {
    struct ncclCommDevMetaSlab* slab = nullptr;
    NCCLCHECKGOTO(ncclCalloc(&slab, 1), ret, exit);

    const int nRanks = comm->nRanks;
    const int nPeers = nRanks + 1 /* CollNet */ + comm->localRanks /* NVLS */;
    slab->nPeersPerChannel = nPeers;

    size_t cursor = 0;
    slab->offCommAndChans = cursor;
    cursor = alignUp(cursor + sizeof(struct ncclKernelCommAndChannels));

    slab->offRankToLocalRank = cursor;
    cursor = alignUp(cursor + sizeof(int) * nRanks);

    if (comm->denseToUserRank != nullptr) {
      slab->offDenseToUserRank = cursor;
      cursor = alignUp(cursor + sizeof(int) * nRanks);
    }

    for (int c = 0; c < MAXCHANNELS; c++) {
      slab->offChanDevPeers[c] = cursor;
      cursor = alignUp(cursor + sizeof(struct ncclDevChannelPeer*) * nPeers);
    }
    for (int c = 0; c < MAXCHANNELS; c++) {
      slab->offChanRingUserRanks[c] = cursor;
      cursor = alignUp(cursor + sizeof(int) * nRanks);
    }
    slab->totalBytes = cursor;

    NCCLCHECKGOTO(ncclCalloc(&slab->hostStaging, slab->totalBytes), ret, fail_slab);
    NCCLCHECKGOTO(ncclCudaCallocAsync(&slab->devBase, slab->totalBytes, deviceStream, comm->memManager, ncclMemOffload),
                  ret, fail_slab);
    ncclCommPushCudaFree(comm, slab->devBase);
    comm->devMetaSlab = slab;
    goto exit;

  fail_slab:
    free(slab->hostStaging);
    free(slab);
  }

exit:
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream,
                                    /*concurrent=*/false));
  return ret;
}

ncclResult_t ncclChannelSlabFlushToDevice(struct ncclComm* comm,
                                          const struct ncclKernelCommAndChannels* commAndChans,
                                          cudaStream_t deviceStream) {
  struct ncclCommDevMetaSlab* slab = comm->devMetaSlab;
  if (slab == nullptr || slab->hostStaging == nullptr) return ncclInternalError;

  const int nRanks = comm->nRanks;
  const int nPeers = slab->nPeersPerChannel;

  memcpy(slab->hostStaging + slab->offCommAndChans, commAndChans, sizeof(*commAndChans));
  memcpy(slab->hostStaging + slab->offRankToLocalRank, comm->rankToLocalRank, sizeof(int) * nRanks);
  if (comm->denseToUserRank != nullptr) {
    memcpy(slab->hostStaging + slab->offDenseToUserRank, comm->denseToUserRank, sizeof(int) * nRanks);
  }

  for (int c = 0; c < MAXCHANNELS; c++) {
    const struct ncclChannel* ch = &comm->channels[c];
    if (ch->id == -1) continue;
    if (ch->devPeersHostPtr != nullptr) {
      memcpy(slab->hostStaging + slab->offChanDevPeers[c], ch->devPeersHostPtr,
             sizeof(struct ncclDevChannelPeer*) * nPeers);
    }
    if (ch->ring.userRanks != nullptr) {
      memcpy(slab->hostStaging + slab->offChanRingUserRanks[c], ch->ring.userRanks, sizeof(int) * nRanks);
    }
  }

  NCCLCHECK(ncclCudaMemcpyAsync(slab->devBase, slab->hostStaging, slab->totalBytes, deviceStream));
  return ncclSuccess;
}

void ncclChannelSlabDestroyHost(struct ncclComm* comm) {
  if (comm->devMetaSlab != nullptr) {
    free(comm->devMetaSlab->hostStaging);
    comm->devMetaSlab->hostStaging = nullptr;
    free(comm->devMetaSlab);
    comm->devMetaSlab = nullptr;
  }
}
