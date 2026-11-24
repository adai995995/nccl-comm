// Regression for current-context reuse and missing-context recovery.
#include "device_env.h"
#include "strongstream.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); abort(); } } while (0)

int main() {
  CHECK(cudaSetDevice(0) == cudaSuccess);
  CHECK(cudaFree(nullptr) == cudaSuccess);
  const ncclDeviceFacts* facts;
  CHECK(ncclDeviceFactsGet(0, &facts) == ncclSuccess);
  CUcontext original = nullptr;
  CHECK(cuCtxGetCurrent(&original) == CUDA_SUCCESS && original != nullptr);
  ncclCudaContext* first = nullptr;
  CHECK(ncclCudaContextTrack(&first, 0, 1, facts) == ncclSuccess);
  CHECK(first->hcontext == original);

  // Detach the thread without destroying its application's primary context.
  CHECK(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS);
  CUcontext detached = original;
  CHECK(cuCtxGetCurrent(&detached) == CUDA_SUCCESS && detached == nullptr);
  ncclCudaContext* second = nullptr;
  CHECK(ncclCudaContextTrack(&second, 0, 2, facts) == ncclSuccess);
  CUcontext restored = nullptr;
  CHECK(cuCtxGetCurrent(&restored) == CUDA_SUCCESS && restored == original);
  CHECK(second == first && second->refCount == 2);
  ncclCudaContextDrop(second);
  ncclCudaContextDrop(first);
  CHECK(cuCtxGetCurrent(&restored) == CUDA_SUCCESS && restored == original);

  // The same fallback must also work after the last ContextEnv reference is gone.
  CHECK(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS);
  CHECK(cuCtxGetCurrent(&detached) == CUDA_SUCCESS && detached == nullptr);
  ncclCudaContext* recreated = nullptr;
  CHECK(ncclCudaContextTrack(&recreated, 0, 3, facts) == ncclSuccess);
  CHECK(recreated->hcontext == original && recreated->facts == facts);
  CHECK(recreated->initState == ncclCudaContext::Ready && recreated->refCount == 1);
  ncclCudaContextDrop(recreated);
  CHECK(cuCtxGetCurrent(&restored) == CUDA_SUCCESS && restored == original);
  puts("PASS: existing context reuse, null-context fallback, release/recreate, application context preserved");
}
