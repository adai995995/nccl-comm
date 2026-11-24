/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_FABRIC_DEVICE_FABRIC_H_
#define NCCL_FABRIC_DEVICE_FABRIC_H_

#include "nccl.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

struct ncclComm;
struct ncclCudaContext;
struct ncclSharedResources;
struct ncclNvlsSharedRes;
struct ncclChannelPeer;
struct ncclDevChannelPeer;
struct ncclFabricAtlas;

struct ncclPhysPeerKey {
  cudaUUID_t gpuUuid;
  uint64_t rawHostHash;
  uint64_t rawPidHash;

  bool operator==(const ncclPhysPeerKey& o) const;
};

int ncclFabricShareEnabled();

// Reserve an exclusive attachment ticket on this context's active DeviceFabric
// before AllGather1. Returns 0 when no active fabric exists or another thread
// on this context is concurrently attaching.
uint64_t ncclDeviceFabricTryReserveTicket(struct ncclComm* comm);

// Release a reserved ticket if init fails before AllGather1 completes.
void ncclDeviceFabricReleaseTicket(struct ncclComm* comm);

// After AllGather1, attach an independent communicator to the active
// DeviceFabric if and only if all ranks unanimously advertised the same
// fabricCandidateId and all physical peers exist in the fabric table.
ncclResult_t ncclDeviceFabricTryBindIndependent(struct ncclComm* comm, bool cuMemHostEnabled);

// Publish a newly initialized root communicator's resources as the context's
// DeviceFabric so subsequent independent or split communicators reuse P2P/QP
// endpoints, Proxy threads, and NVLS multicast groups.
ncclResult_t ncclDeviceFabricPublishRoot(struct ncclComm* comm);

// NVLS clique lookup/registration by physical local peer set + head set.
bool ncclDeviceFabricMatchNvlsClique(struct ncclComm* comm,
                                     struct ncclNvlsSharedRes** outRes,
                                     struct ncclChannelPeer* outOwnerPeers[64],
                                     struct ncclDevChannelPeer* outOwnerDevPeers[64],
                                     std::vector<int>* outSlotMap);
void ncclDeviceFabricRegisterNvlsClique(struct ncclComm* comm, struct ncclNvlsSharedRes* nvlsRes);
void ncclDeviceFabricUnregisterNvlsClique(struct ncclComm* comm, struct ncclNvlsSharedRes* nvlsRes);

int ncclDeviceFabricMaxNvlsChannels(struct ncclComm* comm);

// Retire the active DeviceFabric if an attached communicator aborts so future
// communicators do not attach to an aborted shared transport state, and mark
// existing sibling attachments for channel rebuild before their next launch.
void ncclDeviceFabricRetireOnAbort(struct ncclComm* comm);

// Check whether a surviving communicator's shared fabric was tainted by a
// sibling's ncclCommAbort, and rebuild standalone channels + proxy if needed.
bool ncclDeviceFabricNeedsRebuild(const struct ncclComm* comm);
ncclResult_t ncclDeviceFabricRebuildComm(struct ncclComm* comm);

// Drop comm->boundFabric reference during commFree().
void ncclDeviceFabricReleaseBound(struct ncclComm* comm);

// Called from commDestroySync when sharedRes->refCount reaches 0.
void ncclDeviceFabricOnSharedResDestroy(struct ncclComm* comm, struct ncclSharedResources* sharedRes);

#endif
