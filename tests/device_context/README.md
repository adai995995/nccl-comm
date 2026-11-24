# DeviceFacts / ContextEnv regression checks

Run from the repository root on a Linux CUDA machine. The internal lifecycle
test needs two GPUs, the current-context test needs one GPU, and the process
test defaults to eight GPUs. These check correctness and reuse, not
initialization performance or memory savings.

```sh
make -j32 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
mkdir -p build/test/device_context
g++ -std=c++17 -DNCCL_OS_LINUX \
  -Ibuild/include -Isrc -Isrc/include -Isrc/include/plugin \
  -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \
  tests/device_context/env_lifecycle.cc build/lib/libnccl_static.a \
  -Wl,--wrap=cudaEventCreateWithFlags -L/usr/local/cuda/lib64 \
  -lcudart_static -ldl -lpthread -lrt -o build/test/device_context/env_lifecycle
g++ -std=c++17 -DNCCL_OS_LINUX \
  -Ibuild/include -Isrc -Isrc/include -Isrc/include/plugin \
  -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \
  tests/device_context/context_lazy.cc build/lib/libnccl_static.a \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
  -lcuda -lcudart_static -ldl -lpthread -lrt -o build/test/device_context/context_lazy
g++ -std=c++17 -Ibuild/include -I/usr/local/cuda/include tests/device_context/multi_comm.cc \
  -Lbuild/lib -Wl,-rpath,"$PWD/build/lib" -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lnccl -lcudart -lpthread -o build/test/device_context/multi_comm
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT build/test/device_context/env_lifecycle
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT build/test/device_context/context_lazy
python3 tests/device_context/run_multi.py build/test/device_context/multi_comm 8
```

The internal tests above use the Makefile's default `CUDARTLIB=cudart_static`.
Match their CUDA runtime linkage to the NCCL build if changing that setting.
The driver stub directory is used only at link time; execution uses the installed
NVIDIA driver, so do not add the stub directory to `LD_LIBRARY_PATH`.

The lifecycle test injects a CUDA event creation failure after stream creation,
then checks retry, concurrent first initialization, per-context kernel tables,
and last-reference teardown/recreation. An expected CUDA allocation warning is
printed by this deliberate failure.

The current-context test first acquires an existing CUDA context, explicitly
unbinds it from the calling thread, and checks that ContextEnv restores the same
primary context and reuses the same shared object. After releasing all references,
it repeats the missing-current-context path with a fresh ContextEnv. It verifies
balanced references and that dropping ContextEnv preserves the application-owned
CUDA context. Unlike tests that always call `cudaFree(nullptr)` before acquiring
ContextEnv, this exercises the lazy context initialization branch directly.

The process test runs independent communicators, shared-resource TP/DP splits,
AllReduce result checks, CUDA Graph capture/replay, parent-before-child teardown,
and a new communicator after all old references are released. Its log assertions
require one DeviceFacts probe and two ContextEnv lifetimes per rank. The launcher
sets NCCL debug logging and retains all logs in a fresh temporary directory.
