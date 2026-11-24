# Topology sharing

This implementation splits `ncclTopoGetSystem` into the following stages:

1. `ncclTopoProbeLocal`: reuse XML input parsing and CPU/PCI/GPU/NVLink discovery.
2. `ncclTopoApplyLocalComm` / `ncclTopoApplyNetwork`: apply the current rank,
   effective GDR capability, selected plugins and NIC/vNIC policy to a private XML.
3. `ncclTopoGetNodeSnapshot`: agree on reuse or exchange/fuse the node XML.
4. `ncclTopoBuildCommSystem`: construct a private mutable topology. Existing
   path computation, trimming and Ring/Tree search remain per communicator.

The cache lives inside the library, in process-local CPU memory. It does not
require an application-managed global communicator or a daemon. A large warmup
communicator can populate facts for later TP/DP subsets, but only participants
actually observed through an exchange are known. The cache stores compact,
immutable XML trees, without CUDA/plugin handles or pointers to communicators.
It is released at process exit and can evict old entries earlier. Ordinary
communicator destruction only frees that communicator's working topology.

## Two implementation steps

The first step caches local physical probing and keeps every node XML exchange.
Local keys include GPU identity and topology-file identity, size and modification
timestamps. File changes cause a miss; unsuccessful probes are never published.
Concurrent first probes serialize publication, while callers modify separate XMLs.

The second step caches both per-device contributions and fused node snapshots.
Contributions omit the owning comm rank but retain all other XML attributes,
including the effective network view. When every local participant advertises a
cold node cache via `ncclPeerInfo`, initialization skips the agreement rounds and
warms the cache directly from the single XML exchange. Otherwise, initialization
exchanges a small signature record and then a hit/miss vote. A single miss makes
all ranks execute the original XML exchange; no cache lock is held across
bootstrap calls.

An identical ordered member set reuses a fused snapshot. A subset or reordered
group selects cached contributions and fuses them in its own member order, then
caches that result. This preserves the original GPU order and NIC union. Cached
GPU ranks are local slots, remapped to the new comm's actual ranks on a private
copy. Reuse skips the large XML exchange, not every AllGather.

Network plugin callbacks and vNIC creation retain their existing behavior. Their
current topology contribution is revalidated each time rather than keeping
plugin-owned properties or handles alive. MNNVL clique fusion and injected XML
with multiple rank-bearing GPUs in one contribution use the original exchange.
Physical hardware is assumed stable during cache use; live NVLink/PCI hotplug or
reset recovery is not implemented. Use level 0 when physical reprobes are needed.

## Controls

| Environment setting | Behavior |
| --- | --- |
| `NCCL_TOPO_CACHE=0` | Fresh local probes and original node XML exchange |
| `NCCL_TOPO_CACHE=1` | Local probe reuse, original node XML exchange |
| `NCCL_TOPO_CACHE=2` (default) | Local probe and node snapshot reuse |
| `NCCL_TOPO_CACHE_MAX_BYTES=67108864` (default) | LRU budget for retained snapshot data and entry/key accounting |
| `NCCL_FABRIC_SHARE=1` (default) | Context-scoped `DeviceFabric` reuse across independent and split comms |
| `NCCL_TOPO_ROUTE_MEMO=1` (default) | Slot-normalized `ncclTopoSystem` and Ring/Tree/CollNet/NVLS graph memoization |
| `NCCL_BOOTSTRAP_SUBVIEW=1` (default) | Zero-socket multiplexed subviews for resource-sharing `ncclCommSplit` |
| `NCCL_FAST_GROUP_COMPILE=1` (default) | `FabricAtlas` + `GroupCompiler` local synthesis of `AllGather1`/`AllGather3` on splits |

The budget excludes allocator overhead and snapshots pinned by in-flight readers.
A zero or insufficient budget disables retention; it does not fail comm creation.
Node reuse is enabled only if all local participants advertise level 2 through
the existing peer-info AllGather. Different levels therefore cannot split peers
between incompatible collective sequences. `ncclTopoCacheClear` is an internal
eviction hook used by tests; it does not revoke snapshots already held by readers.

## Build and test

Run from the repository root on a Linux CUDA machine. The integration test needs
eight GPUs by default (or a positive multiple of four). Adjust the architecture
and CUDA installation to match the test machine.

```sh
make -j32 src.build CUDA_HOME=/usr/local/cuda NVCC_GENCODE='-gencode=arch=compute_90,code=sm_90'
mkdir -p build/test/topology_cache
for test in cache_unit multi_comm; do
  g++ -std=c++17 -DNCCL_OS_LINUX \
    -Ibuild/include -Isrc -Isrc/include -Isrc/include/plugin \
    -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl \
    tests/topology_cache/$test.cc build/lib/libnccl_static.a \
    -L/usr/local/cuda/lib64 -lcudart_static -ldl -lpthread -lrt \
    -o build/test/topology_cache/$test
done
build/test/topology_cache/cache_unit
for mode in 0 1 2 mixed tiny; do
  python3 tests/topology_cache/run_multi.py build/test/topology_cache/multi_comm "$mode" 8
done
```

Match the test's CUDA runtime linkage to the NCCL build: these commands assume
the Makefile's default `CUDARTLIB=cudart_static`. The launcher retains raw logs
locally; they contain machine-specific identifiers and should not be committed.

The CPU test covers failed-probe retry, concurrent publication, private copies,
file/device invalidation, serialization and bounded eviction. The GPU test checks
independent comms, reversed ranks, TP/DP, a single-rank cache eviction, changed GDR
input, simulated non-contiguous node-local ranks, concurrent initialization,
CUDA Graph replay, parent-first destruction and
recreation. It compares every XML attribute and child order against a fresh
probe plus the original exchange/fusion, and verifies actual AllGather ordering
and AllReduce results. `mixed` disables caching on one rank; `tiny` forces cache
capacity misses. These are correctness checks, not performance measurements.
