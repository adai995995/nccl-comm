// Internal CPU snapshot tests. Link against this checkout's static NCCL.
#include "graph/topo_cache.h"
#include "graph/topo.h"
#include "graph/slot_route_memo.h"
#include "fabric/group_compiler.h"
#include "comm.h"
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
  setenv("NCCL_TOPO_CACHE", "2", 1);
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
  CHECK(!ncclTopoNodeCacheWarm());
  ncclTopoCacheInsert("local-only", packed);
  CHECK(!ncclTopoNodeCacheWarm());
  ncclTopoCacheInsert("fragment-0", packed);
  CHECK(ncclTopoNodeCacheWarm());
  ncclTopoCacheClear();
  CHECK(!ncclTopoNodeCacheWarm());
  ncclTopoCacheInsert("held", packed);
  auto held = ncclTopoCacheFind("held");
  CHECK(held && *held == packed);
  ncclTopoCacheClear();
  CHECK(!ncclTopoCacheFind("held") && *held == packed);
  ncclTopoCacheInsert("system-0", packed);
  CHECK(ncclTopoNodeCacheWarm());
  for (int i = 0; i < 32; i++) ncclTopoCacheInsert(std::to_string(i), std::string(4096, 'x'));
  CHECK(!ncclTopoCacheFind("0") && ncclTopoCacheFind("31"));
  CHECK(!ncclTopoNodeCacheWarm());
  ncclTopoCacheInsert("oversized", std::string(65536, 'x'));
  CHECK(!ncclTopoCacheFind("oversized"));

  // Verify host/rank/GUID-invariant normalized node digest for SlotRouteMemo & FabricAtlas,
  // while preserving multi-port shared-ASIC equivalence vs dual-ASIC independence.
  uint64_t savedDigest[2] = {0, 0};
  {
    auto buildNodeXml = [](uint64_t hostHash, int r0, int r1, uint64_t guid0, uint64_t guid1,
                           bool extraNic) -> ncclXml* {
      ncclXml* nx = nullptr;
      CHECK(xmlAlloc(&nx, 32) == ncclSuccess);
      ncclXmlNode *sys, *cpu, *pci0, *gpu0, *nic0, *net0, *pci1, *gpu1, *nic1, *net1, *nicExtra;
      CHECK(xmlAddNode(nx, nullptr, "system", &sys) == ncclSuccess);
      CHECK(xmlSetAttrInt(sys, "version", 1) == ncclSuccess);
      CHECK(xmlAddNode(nx, sys, "cpu", &cpu) == ncclSuccess);
      CHECK(xmlSetAttrLong(cpu, "host_hash", (int64_t)hostHash) == ncclSuccess);
      CHECK(xmlAddNode(nx, cpu, "pci", &pci0) == ncclSuccess);
      CHECK(xmlSetAttrLong(pci0, "busid", 0x1000) == ncclSuccess);
      CHECK(xmlAddNode(nx, pci0, "gpu", &gpu0) == ncclSuccess);
      CHECK(xmlSetAttrInt(gpu0, "rank", r0) == ncclSuccess);
      CHECK(xmlSetAttrInt(gpu0, "dev", 0) == ncclSuccess);
      CHECK(xmlAddNode(nx, pci0, "nic", &nic0) == ncclSuccess);
      CHECK(xmlAddNode(nx, nic0, "net", &net0) == ncclSuccess);
      CHECK(xmlSetAttrInt(net0, "dev", 0) == ncclSuccess);
      CHECK(xmlSetAttrUint64(net0, "guid", guid0) == ncclSuccess);
      CHECK(xmlAddNode(nx, cpu, "pci", &pci1) == ncclSuccess);
      CHECK(xmlSetAttrLong(pci1, "busid", 0x2000) == ncclSuccess);
      CHECK(xmlAddNode(nx, pci1, "gpu", &gpu1) == ncclSuccess);
      CHECK(xmlSetAttrInt(gpu1, "rank", r1) == ncclSuccess);
      CHECK(xmlSetAttrInt(gpu1, "dev", 1) == ncclSuccess);
      CHECK(xmlAddNode(nx, pci1, "nic", &nic1) == ncclSuccess);
      CHECK(xmlAddNode(nx, nic1, "net", &net1) == ncclSuccess);
      CHECK(xmlSetAttrInt(net1, "dev", 1) == ncclSuccess);
      CHECK(xmlSetAttrUint64(net1, "guid", guid1) == ncclSuccess);
      if (extraNic) {
        CHECK(xmlAddNode(nx, pci1, "nic", &nicExtra) == ncclSuccess);
      }
      return nx;
    };
    ncclXml* node0 = buildNodeXml(0xAAAABBBBULL, 0, 1, 0x1070fd0300a1b2c0ULL, 0x1070fd0300a1b2c8ULL, false);
    ncclXml* node1 = buildNodeXml(0xCCCCDDDDULL, 8, 9, 0x9080ef0400d4e5f0ULL, 0x9080ef0400d4e5f8ULL, false);
    ncclXml* nodeDualPortSameAsic =
      buildNodeXml(0xCCCCDDDDULL, 8, 9, 0x9080ef0400d4e5f0ULL, 0x9080ef0400d4e5f0ULL, false);
    ncclXml* node1Het = buildNodeXml(0xCCCCDDDDULL, 8, 9, 0x9080ef0400d4e5f0ULL, 0x9080ef0400d4e5f8ULL, true);
    uint64_t d0[2] = {0, 0}, d1[2] = {0, 0}, dSharedAsic[2] = {0, 0}, dHet[2] = {0, 0};
    CHECK(ncclTopoComputeNormalizedNodeDigest(node0, {0, 1}, d0) == ncclSuccess);
    CHECK(ncclTopoComputeNormalizedNodeDigest(node1, {8, 9}, d1) == ncclSuccess);
    CHECK(ncclTopoComputeNormalizedNodeDigest(nodeDualPortSameAsic, {8, 9}, dSharedAsic) == ncclSuccess);
    CHECK(ncclTopoComputeNormalizedNodeDigest(node1Het, {8, 9}, dHet) == ncclSuccess);
    // Distinct IB GUIDs on structurally isomorphic nodes must match; shared-ASIC vs dual-ASIC must differ.
    CHECK(d0[0] == d1[0] && d0[1] == d1[1]);
    CHECK(d0[0] != dSharedAsic[0] || d0[1] != dSharedAsic[1]);
    CHECK(d0[0] != dHet[0] || d0[1] != dHet[1]);
    // Verify original GUID attribute was restored intact after digest calculation.
    uint64_t restoredGuid = 0;
    ncclXmlNode* netNode0 = nullptr;
    CHECK(xmlFindTag(node0, "net", &netNode0) == ncclSuccess && netNode0 != nullptr);
    CHECK(xmlGetAttrUint64Default(netNode0, "guid", &restoredGuid, 0) == ncclSuccess);
    CHECK(restoredGuid == 0x1070fd0300a1b2c0ULL);
    savedDigest[0] = d0[0];
    savedDigest[1] = d0[1];
    free(node0);
    free(node1);
    free(nodeDualPortSameAsic);
    free(node1Het);
  }

  // Verify SlotRouteMemo Store -> TryGetSystem + TryGetGraphs round-trip when CollNet=0 and NVLS=0 (F7).
  {
    setenv("NCCL_TOPO_ROUTE_MEMO", "1", 1);
    ncclSlotRouteMemoClear();

    ncclComm* commA = nullptr;
    ncclComm* commB = nullptr;
    ncclTopoGraph* g = nullptr;
    ncclTopoGraph* outG = nullptr;
    CHECK(ncclCalloc(&commA, 1) == ncclSuccess);
    CHECK(ncclCalloc(&commB, 1) == ncclSuccess);
    CHECK(ncclCalloc(&g, 5) == ncclSuccess);
    CHECK(ncclCalloc(&outG, 5) == ncclSuccess);

    ncclPeerInfo peersA[2]{};
    peersA[0].shmDev = 42;
    peersA[0].pidHash = 1001;
    peersA[1].shmDev = 42;
    peersA[1].pidHash = 1002;
    commA->nRanks = 2;
    commA->peerInfo = peersA;
    commA->config.collnetEnable = 0;
    commA->nvlsSupport = 0;
    commA->minCompCap = commA->maxCompCap = 90;
    CHECK(ncclCalloc(&commA->topo, 1) == ncclSuccess);
    commA->topo->nHosts = 1;
    commA->topo->systemId = 0;
    commA->topo->hostHashes[0] = 0x1234ULL;
    commA->topo->nodes[GPU].count = 2;
    commA->topo->nodes[GPU].nodes[0].type = GPU;
    commA->topo->nodes[GPU].nodes[0].gpu.rank = 0;
    commA->topo->nodes[GPU].nodes[1].type = GPU;
    commA->topo->nodes[GPU].nodes[1].gpu.rank = 1;

    g[0].nChannels = 2;
    g[0].intra[0] = 1;
    g[0].intra[1] = 0;
    g[0].intra[2] = 0;
    g[0].intra[3] = 1;
    ncclTopoGraph* gp[5] = {&g[0], &g[1], &g[2], &g[3], &g[4]};
    CHECK(ncclSlotRouteMemoStore(commA, savedDigest, {0, 1}, gp) == ncclSuccess);

    ncclPeerInfo peersB[10]{};
    peersB[8].shmDev = 99;
    peersB[8].pidHash = 2001;
    peersB[9].shmDev = 99;
    peersB[9].pidHash = 2002;
    commB->nRanks = 2;
    commB->peerInfo = peersB;
    commB->config.collnetEnable = 0;
    commB->nvlsSupport = 0;
    commB->minCompCap = commB->maxCompCap = 90;

    bool sysHit = false, graphsHit = false;
    ncclTopoGraph* outGp[5] = {&outG[0], &outG[1], &outG[2], &outG[3], &outG[4]};
    CHECK(ncclSlotRouteMemoTryGetSystem(commB, savedDigest, {8, 9}, &sysHit) == ncclSuccess && sysHit);
    CHECK(commB->topo != nullptr && commB->topo->nodes[GPU].nodes[0].gpu.rank == 8 &&
          commB->topo->nodes[GPU].nodes[1].gpu.rank == 9);
    CHECK(ncclSlotRouteMemoTryGetGraphs(commB, savedDigest, {8, 9}, outGp, &graphsHit) == ncclSuccess && graphsHit);
    CHECK(outG[0].nChannels == 2 && outG[0].intra[0] == 9 && outG[0].intra[1] == 8 &&
          outG[0].intra[2] == 8 && outG[0].intra[3] == 9);

    ncclTopoFree(commA->topo);
    ncclTopoFree(commB->topo);
    free(outG);
    free(g);
    free(commB);
    free(commA);
    ncclSlotRouteMemoClear();
  }

  // Verify GroupCompiler split policy digest detects per-rank config divergences (F3).
  {
    ncclComm* comms = nullptr;
    CHECK(ncclCalloc(&comms, 4) == ncclSuccess);
    ncclComm &parent = comms[0], &child0 = comms[1], &child1 = comms[2], &child2 = comms[3];
    parent.compCap = 90;
    parent.cudaArch = 900;
    child0.config = NCCL_CONFIG_INITIALIZER;
    child1.config = NCCL_CONFIG_INITIALIZER;
    child2.config = NCCL_CONFIG_INITIALIZER;
    child0.config.maxP2pPeers = 1;
    child1.config.maxP2pPeers = 1;
    child2.config.maxP2pPeers = 2;
    uint64_t p0 = ncclGroupCompilerSplitPolicyDigest(&child0, &parent);
    uint64_t p1 = ncclGroupCompilerSplitPolicyDigest(&child1, &parent);
    uint64_t p2 = ncclGroupCompilerSplitPolicyDigest(&child2, &parent);
    CHECK(p0 != 0 && p0 == p1 && p0 != p2);
    child1.config.nChannelsPerNetPeer = 4;
    CHECK(ncclGroupCompilerSplitPolicyDigest(&child1, &parent) != p0);
    child1.config.nChannelsPerNetPeer = child0.config.nChannelsPerNetPeer;
    setenv("NCCL_NVLS_ENABLE", "0", 1);
    uint64_t pNvls0 = ncclGroupCompilerSplitPolicyDigest(&child0, &parent);
    setenv("NCCL_NVLS_ENABLE", "2", 1);
    uint64_t pNvls2 = ncclGroupCompilerSplitPolicyDigest(&child0, &parent);
    unsetenv("NCCL_NVLS_ENABLE");
    CHECK(pNvls0 != pNvls2);
    parent.nvlsSupport = 1;
    CHECK(ncclGroupCompilerSplitPolicyDigest(&child0, &parent) != p0);
    free(comms);
  }
  puts("PASS: retry, concurrent probe, private XML, file/device invalidation, roundtrip, bounded eviction, normalized digest, memo & policy");
}
