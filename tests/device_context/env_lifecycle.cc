// Internal lifecycle regression test; link against this checkout's static NCCL.
#include "device_env.h"
#include "strongstream.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); abort(); } } while (0)
static std::atomic<bool> failEvent{false};
extern "C" cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
extern "C" cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  if (failEvent.exchange(false)) return cudaErrorMemoryAllocation;
  return __real_cudaEventCreateWithFlags(event, flags);
}

int main() {
  CHECK(cudaSetDevice(0) == cudaSuccess);
  CHECK(cudaFree(nullptr) == cudaSuccess);
  const ncclDeviceFacts *facts, *again;
  CHECK(ncclDeviceFactsGet(0, &facts) == ncclSuccess);
  CHECK(ncclDeviceFactsGet(0, &again) == ncclSuccess && facts == again);

  // Fail after a stream was created: no failed object may escape as Ready.
  ncclCudaContext* failed = nullptr;
  failEvent = true;
  CHECK(ncclCudaContextTrack(&failed, 0, 1, facts) != ncclSuccess);
  CHECK(failed == nullptr && !failEvent.load());

  // A cold concurrent acquire must publish a single fully initialized object.
  constexpr int n = 8;
  std::atomic<int> arrived{0};
  std::atomic<bool> release{false};
  ncclCudaContext* contexts[n] = {};
  std::vector<std::thread> threads;
  for (int i = 0; i < n; ++i) threads.emplace_back([&, i] {
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaFree(nullptr) == cudaSuccess);
    const ncclDeviceFacts* local;
    CHECK(ncclDeviceFactsGet(0, &local) == ncclSuccess && local == facts);
    CHECK(ncclCudaContextTrack(&contexts[i], 0, i + 2, local) == ncclSuccess);
    arrived.fetch_add(1);
    while (!release.load()) std::this_thread::yield();
    ncclCudaContextDrop(contexts[i]);
  });
  while (arrived.load() != n) std::this_thread::yield();
  for (int i = 0; i < n; ++i) {
    CHECK(contexts[i] == contexts[0]);
    CHECK(contexts[i]->initState == ncclCudaContext::Ready);
    CHECK(contexts[i]->kernels.symMaxThreads != nullptr);
  }
  CHECK(contexts[0]->refCount == n);

  // Another device/context has independent kernel result storage.
  int deviceCount;
  CHECK(cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount >= 2);
  CHECK(cudaSetDevice(1) == cudaSuccess);
  CHECK(cudaFree(nullptr) == cudaSuccess);
  const ncclDeviceFacts* otherFacts;
  ncclCudaContext* other;
  CHECK(ncclDeviceFactsGet(1, &otherFacts) == ncclSuccess && otherFacts != facts);
  CHECK(ncclCudaContextTrack(&other, 0, 20, otherFacts) == ncclSuccess);
  CHECK(other != contexts[0]);
  CHECK(other->kernels.symMaxThreads != contexts[0]->kernels.symMaxThreads);
  ncclCudaContextDrop(other);
  release = true;
  for (auto& t : threads) t.join();

  // Last-reference teardown permits a fresh ContextEnv with the same facts.
  CHECK(cudaSetDevice(0) == cudaSuccess);
  ncclCudaContext* recreated;
  CHECK(ncclCudaContextTrack(&recreated, 0, 21, facts) == ncclSuccess);
  CHECK(recreated->initState == ncclCudaContext::Ready && recreated->refCount == 1);
  CHECK(recreated->facts == facts);
  ncclCudaContextDrop(recreated);
  puts("PASS: failure cleanup, concurrent reuse, context isolation, release/recreate");
}
