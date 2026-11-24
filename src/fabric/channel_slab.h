/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_FABRIC_CHANNEL_SLAB_H_
#define NCCL_FABRIC_CHANNEL_SLAB_H_

#include "nccl.h"
#include <cuda_runtime.h>
#include <stddef.h>

struct ncclComm;
struct ncclKernelCommAndChannels;

// Consolidates per-communicator read-only device metadata tables
// (ncclKernelCommAndChannels, rankToLocalRank, denseToUserRank, and all
// channels' devPeers[] + devRingUserRanks[]) into a single device allocation
// and a single host-to-device DMA transfer during devCommSetup().
struct ncclCommDevMetaSlab {
  char* devBase;
  char* hostStaging;
  size_t totalBytes;
  int nPeersPerChannel;
  size_t offCommAndChans;
  size_t offRankToLocalRank;
  size_t offDenseToUserRank;
  size_t offChanDevPeers[64 /*MAXCHANNELS*/];
  size_t offChanRingUserRanks[64 /*MAXCHANNELS*/];
};

ncclResult_t ncclChannelSlabAlloc(struct ncclComm* comm);
ncclResult_t ncclChannelSlabFlushToDevice(struct ncclComm* comm,
                                          const struct ncclKernelCommAndChannels* commAndChans,
                                          cudaStream_t deviceStream);
void ncclChannelSlabDestroyHost(struct ncclComm* comm);

#endif
