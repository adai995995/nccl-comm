// Internal CPU snapshot tests. Link against this checkout's static NCCL.
#include "graph/topo_cache.h"
#include "transport.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); abort(); } } while (0)
static std::atomic<int> probes{0};
static std::atomic<bool> failProbe{false};

static ncclResult_t probe(int64_t busId, const char*, ncclXml* xml) {
  probes++;
  if (failProbe.exchange(false)) return ncclSystemError;
  ncclXmlNode *root, *cpu, *pci, *gpu;
  NCCLCHECK(xmlAddNode(xml, nullptr, "system", &root));
  NCCLCHECK(xmlSetAttrInt(root, "version", 1));
  NCCLCHECK(xmlAddNode(xml, root, "cpu", &cpu));
  NCCLCHECK(xmlSetAttrInt(cpu, "numaid", 0));
  NCCLCHECK(xmlAddNode(xml, cpu, "pci", &pci));
  NCCLCHECK(xmlSetAttrLong(pci, "busid", busId));
  NCCLCHECK(xmlAddNode(xml, pci, "gpu", &gpu));
  NCCLCHECK(xmlSetAttrInt(gpu, "sm", 90));
  return ncclSuccess;
}

int main() {
  setenv("NCCL_TOPO_CACHE", "1", 1);
  setenv("NCCL_TOPO_CACHE_MAX_BYTES", "65536", 1);
  char path[] = "/tmp/nccl-topo-cache-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0);
  close(fd);
  setenv("NCCL_TOPO_FILE", path, 1);
  ncclPeerInfo peer{};
  peer.busId = 0x1000;
  peer.mloPart = -1;
  ncclXml* xml = nullptr;
  failProbe = true;
  CHECK(ncclTopoProbeLocal(&peer, probe, &xml) == ncclSystemError && xml == nullptr);
  std::vector<std::thread> threads;
  std::vector<ncclXml*> copies(8);
  for (int i = 0; i < 8; i++) threads.emplace_back([&, i] {
    CHECK(ncclTopoProbeLocal(&peer, probe, &copies[i]) == ncclSuccess);
  });
  for (auto& t : threads) t.join();
  CHECK(probes == 2); // Failed probes are not published; concurrent success is published once.
  CHECK(xmlSetAttrInt(&copies[0]->nodes[3], "rank", 99) == ncclSuccess);
  const char* rank;
  CHECK(xmlGetAttr(&copies[1]->nodes[3], "rank", &rank) == ncclSuccess && rank == nullptr);
  std::string packed, roundtrip;
  CHECK(ncclTopoPackXml(copies[1], &packed) == ncclSuccess);
  CHECK(ncclTopoUnpackXml(packed, 0, &xml) == ncclSuccess);
  CHECK(xml->maxNodes == 4 && xml->nodes[3].parent == &xml->nodes[2]);
  CHECK(ncclTopoPackXml(xml, &roundtrip) == ncclSuccess && packed == roundtrip);
  free(xml);
  CHECK(ncclTopoUnpackXml(packed.substr(0, packed.size() - 1), 0, &xml) != ncclSuccess && xml == nullptr);
  for (auto* copy : copies) free(copy);

  FILE* file = fopen(path, "w");
  CHECK(file && fputs("changed input", file) >= 0 && fclose(file) == 0);
  CHECK(ncclTopoProbeLocal(&peer, probe, &xml) == ncclSuccess && probes == 3);
  free(xml);
  peer.gpuUuid.bytes[0] = 1;
  CHECK(ncclTopoProbeLocal(&peer, probe, &xml) == ncclSuccess && probes == 4);
  free(xml);
  unlink(path);
  ncclTopoCacheClear();
  ncclTopoCacheInsert("held", packed);
  auto held = ncclTopoCacheFind("held");
  CHECK(held && *held == packed);
  ncclTopoCacheClear();
  CHECK(!ncclTopoCacheFind("held") && *held == packed);
  for (int i = 0; i < 32; i++) ncclTopoCacheInsert(std::to_string(i), std::string(4096, 'x'));
  CHECK(!ncclTopoCacheFind("0") && ncclTopoCacheFind("31"));
  ncclTopoCacheInsert("oversized", std::string(65536, 'x'));
  CHECK(!ncclTopoCacheFind("oversized"));
  puts("PASS: retry, concurrent probe, private XML, file/device invalidation, roundtrip, bounded eviction");
}
