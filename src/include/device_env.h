/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_ENV_H_
#define NCCL_DEVICE_ENV_H_

#include "nccl.h"
#include <cuda_runtime.h>
#include <stdint.h>

// Immutable, process-local facts. No communicator, transport, or CUDA context
// ownership. CUDA device ordinals are stable within this process's visibility.
struct ncclDeviceFacts {
  int cudaDev;
  int nvmlDev;
  int64_t busId;
  cudaUUID_t gpuUuid;
  int compCap;
  int cudaArch;
  int driverVersion;
  int maxSharedMemOptin;
  size_t totalGlobalMem;
  int mloPart;
  int fabricHandleSupport;
  int gdrSupport; // -1: requires the legacy, transport-dependent probe
  bool dmaBufSupport; // Raw device capability, before plugin/policy checks
};

// Returned facts remain valid until process teardown. Failed probes are not cached.
ncclResult_t ncclDeviceFactsGet(int cudaDev, const struct ncclDeviceFacts** out);

// Attributes of kernels loaded/configured in one CUDA context. Owned by the
// reference-counted ncclCudaContext (ContextEnv), not by global kernel tables.
struct ncclKernelInfo {
  size_t maxStackBytes;
  int* symMaxDynamicSmem;
  int* symMaxThreads;
  int* genMaxDynamicSmem;
};

#endif
