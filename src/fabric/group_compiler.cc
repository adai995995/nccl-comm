/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "fabric/group_compiler.h"
#include "graph/topo.h"
#include "graph/topo_cache.h"
#include "graph/slot_route_memo.h"
#include "comm.h"
#include "param.h"
#include <algorithm>
#include <cstring>
#include <vector>

NCCL_PARAM(FastGroupCompile, "FAST_GROUP_COMPILE", 1);
extern int64_t ncclParamP2pPerChannelNetBw();

int ncclFastGroupCompileEnabled() {
  return ncclParamFastGroupCompile() != 0 ? 1 : 0;
}

uint32_t ncclGroupCompilerLocalSplitFlags() {
  if (!ncclFastGroupCompileEnabled()) return 0;
  uint32_t flags = 0;
  if (ncclTopoCacheMode() >= 2 && ncclTopoNodeCacheWarm()) {
    flags |= NCCL_SPLIT_COMPILE_PEERINFO;
    if (ncclSlotRouteMemoEnabled() && ncclGetEnv("NCCL_TOPO_FILE") == nullptr &&
        ncclGetEnv("NCCL_GRAPH_FILE") == nullptr) {
      flags |= NCCL_SPLIT_COMPILE_AG3;
    }
  }
  return flags;
}

uint64_t ncclGroupCompilerSplitPolicyDigest(const struct ncclComm* child, const struct ncclComm* parent) {
  if (child == nullptr || parent == nullptr) return 0;
  uint64_t h[2] = {1, 1};
  const ncclConfig_t& cfg = child->config;
  eatHash(h, &cfg.cgaClusterSize);
  eatHash(h, &cfg.minCTAs);
  eatHash(h, &cfg.maxCTAs);
  eatHash(h, &cfg.splitShare);
  eatHash(h, &cfg.trafficClass);
  eatHash(h, &cfg.collnetEnable);
  eatHash(h, &cfg.CTAPolicy);
  eatHash(h, &cfg.shrinkShare);
  eatHash(h, &cfg.nvlsCTAs);
  eatHash(h, &cfg.nChannelsPerNetPeer);
  eatHash(h, &cfg.nvlinkCentricSched);
  eatHash(h, &cfg.graphUsageMode);
  eatHash(h, &cfg.numRmaCtx);
  eatHash(h, &cfg.maxP2pPeers);
  eatHash(h, &cfg.graphStreamOrdering);
  eatHash(h, &cfg.launchOrderImplicit);
  eatHash(h, &cfg.numRmaSig);
  eatHash(h, &cfg.rmaEagerInit);
  eatHash(h, &cfg.hostCftMode);
  eatHash(h, &cfg.nvlsHostMode);
  if (cfg.netName != nullptr && cfg.netName != NCCL_CONFIG_UNDEF_PTR) {
    eatHash(h, cfg.netName, strlen(cfg.netName));
  }
  eatHash(h, &parent->compCap);
  eatHash(h, &parent->cudaArch);
  eatHash(h, &parent->cpuArch);
  eatHash(h, &parent->cpuVendor);
  eatHash(h, &parent->dmaBufSupport);
  eatHash(h, &parent->nvlsSupport);
  eatHash(h, &parent->nvlsChannels);
  eatHash(h, &parent->collNetChainSupport);
  eatHash(h, &parent->collNetHeadsNum);
  eatHash(h, &parent->collNetSupportMatrix);
  eatHash(h, &parent->isAllNvlink);
  int64_t p2pBw = ncclParamP2pPerChannelNetBw();
  eatHash(h, &p2pBw);

  static const char* const kPolicyEnvs[] = {
    "NCCL_NVLS_ENABLE", "NCCL_NVLS_NCHANNELS", "NCCL_NVLS_CHUNKSIZE",
    "NCCL_NVLSTREE_MAX_CHUNKSIZE", "NCCL_NVLS_FIFO_LAYOUT",
    "NCCL_COLLNET_ENABLE", "NCCL_COLLNET_NODE_THRESHOLD",
    "NCCL_CROSS_NIC", "NCCL_ALGO", "NCCL_PROTO", "NCCL_NTHREADS",
    "NCCL_MAX_NCHANNELS", "NCCL_MIN_NCHANNELS", "NCCL_MIN_NRINGS", "NCCL_MAX_NRINGS",
    "NCCL_MAX_CTAS", "NCCL_MIN_CTAS", "NCCL_CGA_CLUSTER_SIZE", "NCCL_CTA_POLICY",
    "NCCL_UNPACK_DOUBLE_NCHANNELS", "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE",
    "NCCL_P2P_DISABLE", "NCCL_P2P_LEVEL", "NCCL_IGNORE_DISABLED_P2P",
    "NCCL_P2P_MAX_PEERS", "NCCL_MIN_P2P_NCHANNELS", "NCCL_MAX_P2P_NCHANNELS",
    "NCCL_NCHANNELS_PER_NET_PEER", "NCCL_P2P_PER_CHANNEL_NET_BW",
    "NCCL_P2P_PXN_LEVEL", "NCCL_PXN_DISABLE", "NCCL_PXN_C2C", "NCCL_NVB_DISABLE",
    "NCCL_P2P_USE_CUDA_MEMCPY", "NCCL_P2P_READ_ENABLE", "NCCL_P2P_DIRECT_DISABLE",
    "NCCL_SHM_DISABLE", "NCCL_SHM_LOCALITY",
    "NCCL_NET", "NCCL_NET_GDR_LEVEL", "NCCL_NET_GDR_READ", "NCCL_NET_GDR_C2C",
    "NCCL_NET_DISABLE_INTRA", "NCCL_NET_FORCE_MERGE", "NCCL_NET_MERGE_LEVEL",
    "NCCL_NET_MERGE_POLICY", "NCCL_NETDEVS_POLICY",
    "NCCL_DMABUF_ENABLE", "NCCL_GDRCOPY_ENABLE",
    "NCCL_MNNVL_ENABLE", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE"
  };
  for (const char* name : kPolicyEnvs) {
    const char* val = ncclGetEnv(name);
    eatHash(h, name, strlen(name));
    if (val != nullptr) eatHash(h, val, strlen(val));
  }
  uint64_t d = digestHash(h);
  return d != 0 ? d : 1ULL;
}

ncclResult_t ncclGroupCompilerTrySynthesizePeerInfo(struct ncclComm* comm, bool* synthesized) {
  *synthesized = false;
  if (!ncclFastGroupCompileEnabled() || comm == nullptr || comm->isGrow) return ncclSuccess;
  if (comm->sharedRes == nullptr || comm->sharedRes->owner == comm) return ncclSuccess;
  if ((comm->splitCompileFlags & NCCL_SPLIT_COMPILE_PEERINFO) == 0) return ncclSuccess;
  const struct ncclFabricAtlas* atlas = comm->fabricAtlas;
  if (atlas == nullptr || comm->atlasWorldRanks == nullptr) return ncclSuccess;
  if (!atlas->allDistinctPids || !atlas->allTopoCacheV2) return ncclSuccess;

  for (int r = 0; r < comm->nRanks; r++) {
    int wr = comm->atlasWorldRanks[r];
    if (wr < 0 || wr >= atlas->worldSize) return ncclSuccess;
    struct ncclPeerInfo pi = atlas->endpoints[wr].basePeerInfo;
    pi.rank = r;
    pi.hostHash = pi.rawHostHash + comm->commHash;
    pi.pidHash = pi.rawPidHash + comm->commHash;
    pi.topoCacheMode = 2;
    pi.topoCacheWarm = true;
    pi.fabricCandidateId = 0;
    pi.comm = (r == comm->rank) ? comm : nullptr;
    comm->peerInfo[r] = pi;
  }

  *synthesized = true;
  INFO(NCCL_INIT, "GroupCompiler: synthesized peerInfo locally for comm %p nRanks %d (0 AllGather1 packets)",
       comm, comm->nRanks);
  return ncclSuccess;
}

namespace {
ncclResult_t computeRankP2pChannelsPerPeer(struct ncclComm* comm, int targetRank, int* outChannels) {
  int g = 0;
  while (g < comm->topo->nodes[GPU].count && comm->topo->nodes[GPU].nodes[g].gpu.rank != targetRank) g++;
  if (g == comm->topo->nodes[GPU].count) return ncclInternalError;

  int minChannels = MAXCHANNELS;
  struct ncclTopoSystem* system = comm->topo;
  for (int r = 0; r < comm->nRanks; r++) {
    int peerIdx = -1;
    int nCh = -1;
    if (ncclTopoRankToIndex(system, r, &peerIdx, /*showWarn=*/false) == ncclSuccess) {
      if (g == peerIdx) continue;
      struct ncclTopoLinkList* path = system->nodes[GPU].nodes[peerIdx].paths[GPU] + g;
      if (path->type == PATH_NVL || path->type == PATH_NVB) {
        float nvlBw = ncclTopoNVLinkBw(system->nodes[GPU].nodes[g].gpu.cudaCompCap);
        nCh = 2 * std::max(1, (int)(path->bw / nvlBw));
      } else {
        nCh = 2;
      }
    } else {
      int nNetChannels = comm->config.nChannelsPerNetPeer;
      if (nNetChannels == NCCL_CONFIG_UNDEF_INT) {
        float netBw = 0.0f;
        int netCount = 0;
        NCCLCHECK(ncclTopoGetLocalNetCountByBw(system, g, &netCount, &netBw));
        nNetChannels = 2;
        if (netCount > 0) nNetChannels = std::max(netCount, divUp((int)netBw, (int)ncclParamP2pPerChannelNetBw()));
      }
      nCh = nNetChannels;
    }
    if (nCh >= 0) minChannels = std::min(minChannels, nCh);
  }
  *outChannels = minChannels;
  return ncclSuccess;
}

bool fillLocalTopoRanks(int targetRank, int localRanksCount, int nChannels,
                        struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS],
                        struct ncclTopoRanks* outRanks) {
  memset(outRanks, 0, sizeof(*outRanks));
  outRanks->crossNicRing = graphs[NCCL_ALGO_RING]->crossNic;
  outRanks->nvlsHeadNum = 0;

  int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
  int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;
  if (child0Index >= localRanksCount || child1Index >= localRanksCount) return false;

  for (int c = 0; c < nChannels; c++) {
    int* ringIntra = graphs[NCCL_ALGO_RING]->intra + c * localRanksCount;
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra + c * localRanksCount;
    bool foundRing = false, foundTree = false;
    for (int i = 0; i < localRanksCount; i++) {
      if (ringIntra[i] == targetRank) {
        outRanks->ringRecv[c] = ringIntra[0];
        outRanks->ringSend[c] = ringIntra[localRanksCount - 1];
        outRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i - 1];
        outRanks->ringNext[c] = (i == localRanksCount - 1) ? -1 : ringIntra[i + 1];
        foundRing = true;
      }
      if (treeIntra[i] == targetRank) {
        outRanks->treeToParent[c] = treeIntra[0];
        outRanks->treeToChild0[c] = treeIntra[child0Index];
        outRanks->treeToChild1[c] = treeIntra[child1Index];
        foundTree = true;
      }
    }
    if (!foundRing || !foundTree) return false;
  }

  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanksCount;
    bool addHead = true;
    for (int dup = 0; dup < outRanks->nvlsHeadNum; dup++) {
      if (outRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        addHead = false;
        break;
      }
    }
    if (addHead) outRanks->nvlsHeads[outRanks->nvlsHeadNum++] = nvlsIntra[0];
  }
  return true;
}

int remapLocalRank(int r, const std::vector<int>& myLocalRankToPos, const std::vector<int>& dstNodeCommRanks) {
  if (r < 0) return r;
  int pos = myLocalRankToPos[r];
  return dstNodeCommRanks[pos];
}
} // namespace

ncclResult_t ncclGroupCompilerTrySynthesizeAllGather3(struct ncclComm* comm,
                                                      struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS],
                                                      struct ncclAllGather3Info* ag3,
                                                      bool* synthesized) {
  *synthesized = false;
  if (!ncclFastGroupCompileEnabled() || !ncclSlotRouteMemoEnabled() || comm == nullptr) return ncclSuccess;
  if (comm->sharedRes == nullptr || comm->sharedRes->owner == comm) return ncclSuccess;
  if ((comm->splitCompileFlags & NCCL_SPLIT_COMPILE_AG3) == 0) return ncclSuccess;
  if (comm->MNNVL || comm->isGrow || comm->topo == nullptr) return ncclSuccess;
  const struct ncclFabricAtlas* atlas = comm->fabricAtlas;
  if (atlas == nullptr || comm->atlasWorldRanks == nullptr) return ncclSuccess;
  if (!atlas->homogeneousNodes || !atlas->allTopoCacheV2 || !atlas->allDistinctPids) return ncclSuccess;

  // Group comm ranks by physical node in comm rank discovery order, recording
  // each rank's physical nodeLocalSlot sequence.
  std::vector<int> activePhysNodes;
  std::vector<std::vector<int>> nodeCommRanks;
  std::vector<std::vector<int>> nodePhysSlots;
  int mySubNode = -1;

  for (int r = 0; r < comm->nRanks; r++) {
    int wr = comm->atlasWorldRanks[r];
    if (wr < 0 || wr >= atlas->worldSize) return ncclSuccess;
    int physNode = atlas->endpoints[wr].physNodeId;
    int physSlot = atlas->endpoints[wr].nodeLocalSlot;

    int subIdx = -1;
    for (int i = 0; i < int(activePhysNodes.size()); i++) {
      if (activePhysNodes[i] == physNode) {
        subIdx = i;
        break;
      }
    }
    if (subIdx < 0) {
      subIdx = static_cast<int>(activePhysNodes.size());
      activePhysNodes.push_back(physNode);
      nodeCommRanks.emplace_back();
      nodePhysSlots.emplace_back();
    }
    nodeCommRanks[subIdx].push_back(r);
    nodePhysSlots[subIdx].push_back(physSlot);
    if (r == comm->rank) mySubNode = subIdx;
  }

  if (mySubNode < 0) return ncclSuccess;
  const auto& myCommRanks = nodeCommRanks[mySubNode];
  const auto& mySlots = nodePhysSlots[mySubNode];
  int nLocal = static_cast<int>(myCommRanks.size());
  if (nLocal <= 0 || comm->topo->nodes[GPU].count != nLocal) return ncclSuccess;

  // Require every participating node to activate the identical ordered sequence
  // of physical GPU slots (covers TP, DP, PP, CP, EP, and sub-world groups).
  for (size_t n = 0; n < nodePhysSlots.size(); n++) {
    if (nodePhysSlots[n] != mySlots) return ncclSuccess;
  }

  const ncclAllGather3Info selfTemplate = ag3[comm->rank];
  std::vector<int> myLocalRankToPos(comm->nRanks, -1);
  for (int pos = 0; pos < nLocal; pos++) myLocalRankToPos[myCommRanks[pos]] = pos;

  // Step 1: Fill exact entries for all ranks on my own node from comm->topo + graphs.
  std::vector<ncclAllGather3Info> localAg3(nLocal, selfTemplate);
  for (int pos = 0; pos < nLocal; pos++) {
    int r = myCommRanks[pos];
    if (r == comm->rank) {
      localAg3[pos] = selfTemplate;
      continue;
    }
    ncclAllGather3Info& dst = localAg3[pos];
    if (comm->ncclNet && comm->ncclNet->devices) {
      int gpuIdx = -1;
      if (ncclTopoRankToIndex(comm->topo, r, &gpuIdx, false) != ncclSuccess) return ncclSuccess;
      if (ncclTopoGetLocalNetCountByBw(comm->topo, gpuIdx, &dst.localNetCountByBw, &dst.localNetBw) != ncclSuccess)
        return ncclSuccess;
    }
    if (ncclTopoGetMinNetBw(comm->topo, r, &dst.minNetBw) != ncclSuccess) return ncclSuccess;
    if (computeRankP2pChannelsPerPeer(comm, r, &dst.p2pnChannelsPerPeer) != ncclSuccess) return ncclSuccess;
    if (!fillLocalTopoRanks(r, nLocal, comm->nChannels, graphs, &dst.topoRanks)) return ncclSuccess;
  }

  // Step 2: Project localAg3 onto every participating homogeneous node.
  for (size_t n = 0; n < activePhysNodes.size(); n++) {
    const auto& dstRanks = nodeCommRanks[n];
    for (int pos = 0; pos < nLocal; pos++) {
      int dstRank = dstRanks[pos];
      if (int(n) == mySubNode) {
        ag3[dstRank] = localAg3[pos];
        continue;
      }
      ncclAllGather3Info projected = localAg3[pos];
      struct ncclTopoRanks& tr = projected.topoRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        tr.ringRecv[c] = remapLocalRank(tr.ringRecv[c], myLocalRankToPos, dstRanks);
        tr.ringSend[c] = remapLocalRank(tr.ringSend[c], myLocalRankToPos, dstRanks);
        tr.ringPrev[c] = remapLocalRank(tr.ringPrev[c], myLocalRankToPos, dstRanks);
        tr.ringNext[c] = remapLocalRank(tr.ringNext[c], myLocalRankToPos, dstRanks);
        tr.treeToParent[c] = remapLocalRank(tr.treeToParent[c], myLocalRankToPos, dstRanks);
        tr.treeToChild0[c] = remapLocalRank(tr.treeToChild0[c], myLocalRankToPos, dstRanks);
        tr.treeToChild1[c] = remapLocalRank(tr.treeToChild1[c], myLocalRankToPos, dstRanks);
      }
      for (int h = 0; h < tr.nvlsHeadNum; h++) {
        tr.nvlsHeads[h] = remapLocalRank(tr.nvlsHeads[h], myLocalRankToPos, dstRanks);
      }
      ag3[dstRank] = projected;
    }
  }

  *synthesized = true;
  INFO(NCCL_INIT, "GroupCompiler: synthesized AllGather3 locally for comm %p nRanks %d nNodes %zu (0 AllGather3 packets)",
       comm, comm->nRanks, activePhysNodes.size());
  return ncclSuccess;
}
