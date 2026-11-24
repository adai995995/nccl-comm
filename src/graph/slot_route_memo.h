/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GRAPH_SLOT_ROUTE_MEMO_H_
#define NCCL_GRAPH_SLOT_ROUTE_MEMO_H_

#include <stdint.h>
#include <string.h>
#include <vector>
#include "nccl.h"
#include "graph.h"

struct ncclComm;
struct ncclXml;

int ncclSlotRouteMemoEnabled();
void ncclSlotRouteMemoClear();

// Compute a host-agnostic, rank-normalized 128-bit structural digest of an
// assembled node XML (GPU ranks canonicalized to local slots 0..nLocalRanks-1,
// CPU host_hash canonicalized to 0). Homogeneous nodes produce identical digests.
ncclResult_t ncclTopoComputeNormalizedNodeDigest(struct ncclXml* xml, const std::vector<int>& localRanks,
                                                 uint64_t outDigest[2]);

// Try to materialize comm->topo from a cached canonical ncclTopoSystem for
// this node's ordered local subset and policy. Sets *hit = true on success.
ncclResult_t ncclSlotRouteMemoTryGetSystem(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                           const std::vector<int>& localRanks, bool* hit);

// Try to materialize the 5 topology search graphs (remapped from local slots
// 0..nLocalRanks-1 to comm ranks) on a memo hit.
ncclResult_t ncclSlotRouteMemoTryGetGraphs(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                           const std::vector<int>& localRanks,
                                           struct ncclTopoGraph* graphs[5], bool* hit);

// Record a freshly computed system + 5 searched graphs in slot-normalized form.
ncclResult_t ncclSlotRouteMemoStore(struct ncclComm* comm, const uint64_t nodeDigest[2],
                                    const std::vector<int>& localRanks,
                                    struct ncclTopoGraph* const graphs[5]);

#endif
