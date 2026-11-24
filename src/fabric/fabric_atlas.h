/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_FABRIC_ATLAS_H_
#define NCCL_FABRIC_ATLAS_H_

#include <stdint.h>
#include <string.h>
#include <vector>
#include "nccl.h"
#include "graph.h"
#include "transport.h"

struct ncclComm;

struct ncclGraphSummaryInfo {
  int pattern;
  int nChannels;
  int sameChannels;
  float bwIntra;
  float bwInter;
  int typeIntra;
  int typeInter;
  int crossNic;
};

struct ncclAllGather3Info {
  struct ncclGraphSummaryInfo graphInfo[NCCL_NUM_ALGORITHMS];
  struct ncclTopoRanks topoRanks;
  int cpuArch;
  int cpuVendor;
  int localRanks;
  int p2pnChannelsPerPeer;
  int p2pMaxPeers;
  float minNetBw;
  int localNetDeviceCount;
  int localNetCountByBw;
  float localNetBw;
  int localCollNetCount;
  int isAllNvlink;
  uint64_t topoNodeDigest[2];
};

struct ncclAtlasEndpoint {
  struct ncclPeerInfo basePeerInfo;
  int physNodeId;
  int nodeLocalSlot;
  uint64_t nodeTopoDigest[2];
};

// Reference-counted, immutable cluster-wide physical endpoint snapshot captured
// from a root communicator and inherited across split/compiled subgroups.
struct ncclFabricAtlas {
  int refCount = 1;
  uint64_t atlasId = 0;
  uint64_t epoch = 1;
  int worldSize = 0;
  int numPhysNodes = 0;
  int ranksPerNode = 0;
  bool allDistinctPids = false;
  bool allTopoCacheV2 = false;
  bool homogeneousNodes = false;
  uint64_t canonicalNodeDigest[2] = {0, 0};
  std::vector<ncclAtlasEndpoint> endpoints;
};

struct ncclFabricAtlas* ncclFabricAtlasRetain(struct ncclFabricAtlas* atlas);
void ncclFabricAtlasRelease(struct ncclFabricAtlas* atlas);

ncclResult_t ncclFabricAtlasBuildFromRoot(struct ncclComm* comm, const struct ncclAllGather3Info* ag3);
ncclResult_t ncclFabricAtlasInheritChild(struct ncclComm* child, struct ncclComm* parent, const int* parentRanks);

#endif
