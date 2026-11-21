// One process per GPU. Internal topology checks plus real collectives.
#include <nccl.h>
#include "graph/topo_cache.h"
#include "transport.h"
#include <numeric>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define NCCL(call) do { ncclResult_t e = (call); if (e != ncclSuccess) { \
  fprintf(stderr, "NCCL line %d: %s\n", __LINE__, ncclGetErrorString(e)); exit(1); } } while (0)

static ncclUniqueId getId(const std::string& dir, int rank, int wave) {
  std::string path = dir + "/id-" + std::to_string(wave);
  if (rank == 0) {
    ncclUniqueId id;
    NCCL(ncclGetUniqueId(&id));
    std::ofstream f(path + ".tmp", std::ios::binary);
    f.write(reinterpret_cast<char*>(&id), sizeof(id));
    f.close();
    if (!f || std::rename((path + ".tmp").c_str(), path.c_str()) != 0) exit(1);
  }
  for (int i = 0; i < 3000; ++i) {
    ncclUniqueId id;
    std::ifstream f(path, std::ios::binary);
    if (f.read(reinterpret_cast<char*>(&id), sizeof(id))) return id;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  fprintf(stderr, "Timed out waiting for %s\n", path.c_str());
  exit(1);
}

static void reduce(ncclComm_t comm, bool graph) {
  constexpr int count = 4096;
  int nranks;
  NCCL(ncclCommCount(comm, &nranks));
  std::vector<float> input(count, 1.0f), output(count);
  float *send, *recv;
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA(cudaMalloc(&send, count * sizeof(float)));
  CUDA(cudaMalloc(&recv, count * sizeof(float)));
  CUDA(cudaMemcpyAsync(send, input.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
  NCCL(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
  CUDA(cudaStreamSynchronize(stream));
  if (graph) {
    cudaGraph_t g;
    cudaGraphExec_t exec;
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    NCCL(ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream));
    CUDA(cudaStreamEndCapture(stream, &g));
    CUDA(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    CUDA(cudaGraphLaunch(exec, stream));
    CUDA(cudaGraphLaunch(exec, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(g));
  }
  CUDA(cudaMemcpy(output.data(), recv, count * sizeof(float), cudaMemcpyDeviceToHost));
  for (float x : output) if (x != float(nranks)) {
    fprintf(stderr, "Bad AllReduce: %f expected %d\n", x, nranks); exit(1);
  }
  CUDA(cudaFree(send));
  CUDA(cudaFree(recv));
  CUDA(cudaStreamDestroy(stream));
}

static void phase(const char* name) { printf("PHASE %s\n", name); }

static void members(ncclComm_t comm, int device, const std::vector<int>& expected) {
  int* buffer;
  CUDA(cudaMalloc(&buffer, sizeof(int) * (expected.size() + 1)));
  CUDA(cudaMemcpy(buffer, &device, sizeof(int), cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  NCCL(ncclAllGather(buffer, buffer + 1, 1, ncclInt, comm, stream));
  CUDA(cudaStreamSynchronize(stream));
  std::vector<int> actual(expected.size());
  CUDA(cudaMemcpy(actual.data(), buffer + 1, actual.size() * sizeof(int), cudaMemcpyDeviceToHost));
  if (actual != expected) { fprintf(stderr, "Incorrect comm rank mapping\n"); exit(1); }
  CUDA(cudaStreamDestroy(stream));
  CUDA(cudaFree(buffer));
}

static void compareXml(ncclComm_t comm, bool changeGdr = false) {
  int oldGdr = comm->peerInfo[comm->rank].gdrSupport;
  if (changeGdr && comm->rank == 0) comm->peerInfo[comm->rank].gdrSupport = !oldGdr;
  ncclXml* xml;
  std::string cached, fresh;
  NCCL(ncclTopoGetCommXml(comm, nullptr, &xml));
  NCCL(ncclTopoPackXml(xml, &cached));
  free(xml);
  // Force a fresh physical probe and the original node exchange/fusion. The
  // comparison includes every XML attribute and child order before conversion.
  std::vector<int> modes(comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    modes[i] = comm->peerInfo[i].topoCacheMode;
    comm->peerInfo[i].topoCacheMode = 0;
  }
  ncclTopoCacheClear();
  NCCL(ncclTopoGetCommXml(comm, nullptr, &xml));
  NCCL(ncclTopoPackXml(xml, &fresh));
  free(xml);
  for (int i = 0; i < comm->nRanks; i++) comm->peerInfo[i].topoCacheMode = modes[i];
  comm->peerInfo[comm->rank].gdrSupport = oldGdr;
  if (cached != fresh) { fprintf(stderr, "Cached XML differs from fresh topology\n"); exit(1); }
  puts("PASS exact XML equivalence");
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  setvbuf(stdout, nullptr, _IONBF, 0);
  int rank = std::atoi(argv[1]), nranks = std::atoi(argv[2]);
  if (nranks < 4 || nranks % 4) return 2;
  CUDA(cudaSetDevice(rank));
  std::vector<int> all(nranks), reverse(nranks), tpRanks, dpRanks;
  std::iota(all.begin(), all.end(), 0);
  for (int i = 0; i < nranks; i++) {
    reverse[i] = nranks - 1 - i;
    if (i / 4 == rank / 4) tpRanks.push_back(i);
    if (i % 4 == rank % 4) dpRanks.push_back(i);
  }
  ncclComm_t world, independent, reordered, tp, dp, partial, recreated;
  phase("cold");
  NCCL(ncclCommInitRank(&world, nranks, getId(argv[3], rank, 0), rank));
  phase("independent");
  NCCL(ncclCommInitRank(&independent, nranks, getId(argv[3], rank, 1), rank));
  phase("reordered");
  NCCL(ncclCommInitRank(&reordered, nranks, getId(argv[3], rank, 2), nranks - 1 - rank));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;
  phase("tp");
  NCCL(ncclCommSplit(world, rank / 4, rank, &tp, &config));
  phase("dp");
  NCCL(ncclCommSplit(world, rank % 4, rank, &dp, &config));
  phase("collectives");
  members(world, rank, all);
  members(reordered, rank, reverse);
  members(tp, rank, tpRanks);
  members(dp, rank, dpRanks);
  reduce(independent, true);
  phase("strided-local");
  // Exercise non-contiguous node-local ranks without requiring a second host.
  // Only the temporary discovery view changes; the live comm topology is intact.
  std::vector<uint64_t> hosts(nranks);
  for (int i = 0; i < nranks; i++) {
    hosts[i] = world->peerInfo[i].hostHash;
    world->peerInfo[i].hostHash += i % 2;
  }
  compareXml(world);
  for (int i = 0; i < nranks; i++) world->peerInfo[i].hostHash = hosts[i];
  ncclXml* warmed;
  NCCL(ncclTopoGetCommXml(world, nullptr, &warmed));
  free(warmed);
  phase("partial-miss");
  if (rank == 0) ncclTopoCacheClear();
  NCCL(ncclCommInitRank(&partial, nranks, getId(argv[3], rank, 3), rank));
  members(partial, rank, all);
  NCCL(ncclCommDestroy(partial));
  phase("xml-world"); compareXml(world);
  phase("xml-reordered"); compareXml(reordered);
  phase("xml-tp"); compareXml(tp);
  phase("xml-dp"); compareXml(dp);
  phase("changed-gdr"); compareXml(independent, true);
  phase("concurrent");
  ncclUniqueId ids[2] = {getId(argv[3], rank, 4), getId(argv[3], rank, 5)};
  ncclComm_t simultaneous[2];
  std::thread threads[2];
  for (int i = 0; i < 2; i++) threads[i] = std::thread([&, i] {
    CUDA(cudaSetDevice(rank));
    NCCL(ncclCommInitRank(&simultaneous[i], nranks, ids[i], rank));
  });
  for (auto& thread : threads) thread.join();
  for (auto comm : simultaneous) { members(comm, rank, all); NCCL(ncclCommDestroy(comm)); }
  phase("parent-destroy");
  NCCL(ncclCommDestroy(world));
  members(tp, rank, tpRanks);
  members(dp, rank, dpRanks);
  NCCL(ncclCommDestroy(tp));
  NCCL(ncclCommDestroy(dp));
  NCCL(ncclCommDestroy(reordered));
  NCCL(ncclCommDestroy(independent));
  phase("recreated");
  NCCL(ncclCommInitRank(&recreated, nranks, getId(argv[3], rank, 6), rank));
  members(recreated, rank, all);
  NCCL(ncclCommDestroy(recreated));
  printf("PASS rank %d: topology reuse, ordered groups, collective fallback, concurrent init and teardown\n", rank);
}
