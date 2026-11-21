// One process per GPU, no MPI dependency. See README.md for the launcher.
#include <nccl.h>
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

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  int rank = std::atoi(argv[1]), nranks = std::atoi(argv[2]);
  CUDA(cudaSetDevice(rank));
  ncclComm_t world, independent, tp, dp, recreated;
  NCCL(ncclCommInitRank(&world, nranks, getId(argv[3], rank, 0), rank));
  NCCL(ncclCommInitRank(&independent, nranks, getId(argv[3], rank, 1), rank));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;
  NCCL(ncclCommSplit(world, rank / 4, rank, &tp, &config));
  NCCL(ncclCommSplit(world, rank % 4, rank, &dp, &config));
  reduce(world, false);
  reduce(independent, true);
  reduce(tp, false);
  reduce(dp, false);
  NCCL(ncclCommDestroy(world));
  reduce(tp, false);
  reduce(dp, false);
  NCCL(ncclCommDestroy(tp));
  NCCL(ncclCommDestroy(dp));
  NCCL(ncclCommDestroy(independent));
  NCCL(ncclCommInitRank(&recreated, nranks, getId(argv[3], rank, 2), rank));
  reduce(recreated, false);
  NCCL(ncclCommDestroy(recreated));
  printf("PASS rank %d: independent comm, TP/DP, graph replay, parent teardown, recreate\n", rank);
}
