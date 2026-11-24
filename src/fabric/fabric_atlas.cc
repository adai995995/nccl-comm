/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "fabric/fabric_atlas.h"
#include "comm.h"
#include <cstring>
#include <memory>
#include <vector>

struct ncclFabricAtlas* ncclFabricAtlasRetain(struct ncclFabricAtlas* atlas) {
  if (atlas != nullptr) ncclAtomicRefCountIncrement(&atlas->refCount);
  return atlas;
}

void ncclFabricAtlasRelease(struct ncclFabricAtlas* atlas) {
  if (atlas != nullptr && ncclAtomicRefCountDecrement(&atlas->refCount) == 0) {
    delete atlas;
  }
}

ncclResult_t ncclFabricAtlasBuildFromRoot(struct ncclComm* comm, const struct ncclAllGather3Info* ag3) {
  if (comm == nullptr || ag3 == nullptr || comm->MNNVL || comm->isGrow || comm->fabricAtlas != nullptr) {
    return ncclSuccess;
  }

  std::unique_ptr<ncclFabricAtlas> atlas(new (std::nothrow) ncclFabricAtlas{});
  if (!atlas) return ncclSystemError;

  const int W = comm->nRanks;
  atlas->atlasId = comm->commHash != 0 ? comm->commHash : 1ULL;
  atlas->worldSize = W;
  atlas->endpoints.resize(W);

  // Partition world ranks into ordered physical hosts by rawHostHash.
  std::vector<uint64_t> nodeHosts;
  std::vector<int> nodeCounts;
  atlas->allDistinctPids = true;
  atlas->allTopoCacheV2 = true;

  for (int r = 0; r < W; r++) {
    const struct ncclPeerInfo& pi = comm->peerInfo[r];
    if (pi.topoCacheMode < 2) atlas->allTopoCacheV2 = false;

    int nodeId = -1;
    for (int n = 0; n < int(nodeHosts.size()); n++) {
      if (nodeHosts[n] == pi.rawHostHash) {
        nodeId = n;
        break;
      }
    }
    if (nodeId < 0) {
      nodeId = static_cast<int>(nodeHosts.size());
      nodeHosts.push_back(pi.rawHostHash);
      nodeCounts.push_back(0);
    }
    int slot = nodeCounts[nodeId]++;

    for (int prev = 0; prev < r && atlas->allDistinctPids; prev++) {
      if (comm->peerInfo[prev].rawHostHash == pi.rawHostHash &&
          comm->peerInfo[prev].rawPidHash == pi.rawPidHash) {
        atlas->allDistinctPids = false;
      }
    }

    ncclAtlasEndpoint& ep = atlas->endpoints[r];
    ep.basePeerInfo = pi;
    ep.basePeerInfo.comm = nullptr;
    ep.basePeerInfo.fabricCandidateId = 0;
    ep.physNodeId = nodeId;
    ep.nodeLocalSlot = slot;
    ep.nodeTopoDigest[0] = ag3[r].topoNodeDigest[0];
    ep.nodeTopoDigest[1] = ag3[r].topoNodeDigest[1];
  }

  atlas->numPhysNodes = static_cast<int>(nodeHosts.size());
  atlas->ranksPerNode = nodeCounts.empty() ? 0 : nodeCounts[0];
  atlas->canonicalNodeDigest[0] = atlas->endpoints[0].nodeTopoDigest[0];
  atlas->canonicalNodeDigest[1] = atlas->endpoints[0].nodeTopoDigest[1];

  bool homo = (atlas->canonicalNodeDigest[0] != 0 || atlas->canonicalNodeDigest[1] != 0) && atlas->ranksPerNode > 0;
  for (int n = 0; n < atlas->numPhysNodes && homo; n++) {
    if (nodeCounts[n] != atlas->ranksPerNode) homo = false;
  }
  for (int r = 0; r < W && homo; r++) {
    if (atlas->endpoints[r].nodeTopoDigest[0] != atlas->canonicalNodeDigest[0] ||
        atlas->endpoints[r].nodeTopoDigest[1] != atlas->canonicalNodeDigest[1]) {
      homo = false;
    }
  }
  atlas->homogeneousNodes = homo;

  NCCLCHECK(ncclCalloc(&comm->atlasWorldRanks, W));
  for (int r = 0; r < W; r++) comm->atlasWorldRanks[r] = r;
  comm->fabricAtlas = atlas.release();

  INFO(NCCL_INIT, "FabricAtlas: built epoch %lu worldSize %d nodes %d homo %d distinctPids %d",
       (unsigned long)comm->fabricAtlas->epoch, W, comm->fabricAtlas->numPhysNodes,
       homo ? 1 : 0, comm->fabricAtlas->allDistinctPids ? 1 : 0);
  return ncclSuccess;
}

ncclResult_t ncclFabricAtlasInheritChild(struct ncclComm* child, struct ncclComm* parent, const int* parentRanks) {
  if (child == nullptr || parent == nullptr || parentRanks == nullptr || parent->fabricAtlas == nullptr) {
    return ncclSuccess;
  }
  if (child->isGrow || parent->MNNVL) return ncclSuccess;

  NCCLCHECK(ncclCalloc(&child->atlasWorldRanks, child->nRanks));
  for (int r = 0; r < child->nRanks; r++) {
    int pr = parentRanks[r];
    if (pr < 0 || pr >= parent->nRanks) {
      free(child->atlasWorldRanks);
      child->atlasWorldRanks = nullptr;
      return ncclInternalError;
    }
    child->atlasWorldRanks[r] = parent->atlasWorldRanks ? parent->atlasWorldRanks[pr] : pr;
  }
  child->fabricAtlas = ncclFabricAtlasRetain(parent->fabricAtlas);
  return ncclSuccess;
}
