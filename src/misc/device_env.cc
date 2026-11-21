/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device_env.h"
#include "cudawrap.h"
#include "nvmlwrap.h"
#include "utils.h"
#include "graph/topo.h"
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace {
struct DeviceFactsEntry {
  ncclDeviceFacts facts;
  DeviceFactsEntry* next;
};

struct DeviceFactsCache {
  std::mutex mutex;
  DeviceFactsEntry* head = nullptr;
  ~DeviceFactsCache() {
    while (head) {
      DeviceFactsEntry* next = head->next;
      delete head;
      head = next;
    }
  }
};

ncclResult_t probeDeviceFacts(int cudaDev, ncclDeviceFacts* facts) {
  NCCLCHECK(ncclCudaLibraryInit());
  facts->cudaDev = cudaDev;
  cudaDeviceProp prop;
  CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
  facts->compCap = prop.major * 10 + prop.minor;
  facts->cudaArch = facts->compCap * 10;
  facts->totalGlobalMem = prop.totalGlobalMem;
  const char* mloPart = strstr(prop.name, "MLOPart");
  facts->mloPart = mloPart ? atoi(mloPart + strlen("MLOPart")) : NCCL_TOPO_UNDEF;
  CUDACHECK(cudaDeviceGetAttribute(&facts->maxSharedMemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev));
  NCCLCHECK(ncclCudaDriverVersion(&facts->driverVersion));

  NCCLCHECK(getBusId(cudaDev, &facts->busId));
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  nvmlDevice_t nvmlDev;
  NCCLCHECK(int64ToBusId(facts->busId, busId));
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&facts->nvmlDev));
  CUdevice dev;
  CUCHECK(cuDeviceGet(&dev, cudaDev));
  CUCHECK(cuDeviceGetUuid((CUuuid*)&facts->gpuUuid, dev));
  (void)CUPFN(cuDeviceGetAttribute)(&facts->fabricHandleSupport,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
  facts->gdrSupport = -1;
#if CUDART_VERSION >= 11030
  if (facts->driverVersion >= 11030) {
    CUDACHECK(cudaDeviceGetAttribute(&facts->gdrSupport, cudaDevAttrGPUDirectRDMASupported, cudaDev));
  }
#endif
#if CUDA_VERSION >= 11070
  if (facts->driverVersion >= 11070) {
    int supported = 0;
    (void)CUPFN(cuDeviceGetAttribute)(&supported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    facts->dmaBufSupport = supported != 0;
  }
#endif
  return ncclSuccess;
}
} // namespace

ncclResult_t ncclDeviceFactsGet(int cudaDev, const ncclDeviceFacts** out) {
  *out = nullptr;
  static DeviceFactsCache cache;
  std::lock_guard<std::mutex> lock(cache.mutex);
  for (DeviceFactsEntry* entry = cache.head; entry; entry = entry->next) {
    if (entry->facts.cudaDev == cudaDev) {
      *out = &entry->facts;
      return ncclSuccess;
    }
  }
  std::unique_ptr<DeviceFactsEntry> entry(new (std::nothrow) DeviceFactsEntry{});
  if (!entry) return ncclSystemError;
  NCCLCHECK(probeDeviceFacts(cudaDev, &entry->facts));
  entry->next = cache.head;
  cache.head = entry.release();
  *out = &cache.head->facts;
  INFO(NCCL_INIT, "DeviceFacts: probed cudaDev %d busId %lx", cudaDev, (unsigned long)(*out)->busId);
  return ncclSuccess;
}
