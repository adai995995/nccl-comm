/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_FABRIC_GROUP_COMPILER_H_
#define NCCL_FABRIC_GROUP_COMPILER_H_

#include "nccl.h"
#include "fabric/fabric_atlas.h"

struct ncclComm;
struct ncclTopoGraph;

static constexpr uint32_t NCCL_SPLIT_COMPILE_PEERINFO = 0x1u;
static constexpr uint32_t NCCL_SPLIT_COMPILE_AG3 = 0x2u;

int ncclFastGroupCompileEnabled();
uint32_t ncclGroupCompilerLocalSplitFlags();
uint64_t ncclGroupCompilerSplitPolicyDigest(const struct ncclComm* child, const struct ncclComm* parent);

// Synthesize comm->peerInfo[0..nRanks-1] in pure CPU memory from FabricAtlas,
// bypassing fillInfo() driver calls and AllGather1 network exchange.
ncclResult_t ncclGroupCompilerTrySynthesizePeerInfo(struct ncclComm* comm, bool* synthesized);

// Synthesize allGather3Data[0..nRanks-1] deterministically in CPU memory when
// all participating nodes in FabricAtlas share identical topology templates
// and symmetric active slot layouts, bypassing AllGather3 network exchange.
ncclResult_t ncclGroupCompilerTrySynthesizeAllGather3(struct ncclComm* comm,
                                                      struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS],
                                                      struct ncclAllGather3Info* ag3,
                                                      bool* synthesized);

#endif
