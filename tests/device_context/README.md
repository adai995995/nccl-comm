# DeviceFacts / ContextEnv regression checks

Run from the repository root on a Linux CUDA machine. The internal lifecycle
test needs two GPUs; the process test defaults to eight GPUs. These check
correctness and reuse, not initialization performance or memory savings.

```sh
make -j32 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
mkdir -p build/test/device_context
g++ -std=c++17 -DNCCL_OS_LINUX -Ibuild/include -Isrc/include -I/usr/local/cuda/include \
  tests/device_context/env_lifecycle.cc build/lib/libnccl_static.a \
  -Wl,--wrap=cudaEventCreateWithFlags -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lcudart -ldl -lpthread -lrt -o build/test/device_context/env_lifecycle
g++ -std=c++17 -Ibuild/include -I/usr/local/cuda/include tests/device_context/multi_comm.cc \
  -Lbuild/lib -Wl,-rpath,"$PWD/build/lib" -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lnccl -lcudart -lpthread -o build/test/device_context/multi_comm
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT build/test/device_context/env_lifecycle
python3 tests/device_context/run_multi.py build/test/device_context/multi_comm 8
```

The internal test injects a CUDA event creation failure after stream creation,
then checks retry, concurrent first initialization, per-context kernel tables,
and last-reference teardown/recreation. An expected CUDA allocation warning is
printed by this deliberate failure.

The process test runs independent communicators, shared-resource TP/DP splits,
AllReduce result checks, CUDA Graph capture/replay, parent-before-child teardown,
and a new communicator after all old references are released. Its log assertions
require one DeviceFacts probe and two ContextEnv lifetimes per rank. The launcher
sets NCCL debug logging and retains all logs in a fresh temporary directory.
