/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "topo_cache.h"
#include "topo.h"
#include "transport.h"
#include "param.h"
#include <algorithm>
#include <list>
#include <mutex>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

NCCL_PARAM(TopoCache, "TOPO_CACHE", 2);
NCCL_PARAM(TopoCacheMaxBytes, "TOPO_CACHE_MAX_BYTES", 64 * 1024 * 1024);

namespace {
constexpr uint32_t snapshotVersion = 1;

template <typename T>
void append(std::string* out, const T& value) {
  out->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

void appendString(std::string* out, const char* value) {
  uint32_t len = strlen(value);
  append(out, len);
  out->append(value, len);
}

void packNode(const ncclXmlNode* node, std::string* out, uint32_t* count) {
  ++*count;
  appendString(out, node->name);
  append(out, node->type);
  append(out, node->nAttrs);
  for (int a = 0; a < node->nAttrs; a++) {
    appendString(out, node->attrs[a].key);
    appendString(out, node->attrs[a].value);
  }
  append(out, node->nSubs);
  for (int s = 0; s < node->nSubs; s++) packNode(node->subs[s], out, count);
}

struct Reader {
  const std::string& data;
  size_t pos = 0;
  template <typename T> bool get(T* value) {
    if (data.size() - pos < sizeof(T)) return false;
    memcpy(value, data.data() + pos, sizeof(T));
    pos += sizeof(T);
    return true;
  }
  bool string(char* value) {
    uint32_t len;
    if (!get(&len) || len > MAX_STR_LEN || data.size() - pos < len) return false;
    memcpy(value, data.data() + pos, len);
    value[len] = '\0';
    pos += len;
    return true;
  }
};

ncclResult_t unpackNode(Reader* in, ncclXml* xml, ncclXmlNode* parent) {
  char name[MAX_STR_LEN + 1];
  int type, nAttrs, nSubs;
  if (!in->string(name) || !in->get(&type) || !in->get(&nAttrs) || nAttrs < 0 || nAttrs > MAX_ATTR_COUNT)
    return ncclInternalError;
  ncclXmlNode* node;
  NCCLCHECK(xmlAddNode(xml, parent, name, &node));
  node->type = type;
  node->nAttrs = nAttrs;
  for (int a = 0; a < nAttrs; a++) {
    if (!in->string(node->attrs[a].key) || !in->string(node->attrs[a].value)) return ncclInternalError;
  }
  if (!in->get(&nSubs) || nSubs < 0 || nSubs > MAX_SUBS) return ncclInternalError;
  for (int s = 0; s < nSubs; s++) NCCLCHECK(unpackNode(in, xml, node));
  return ncclSuccess;
}


struct Entry {
  std::string key;
  ncclTopoSnapshot data;
  size_t bytes;
};

struct Cache {
  std::mutex mutex;
  std::list<Entry> entries; // Most recently used first.
  std::unordered_map<std::string, std::list<Entry>::iterator> index;
  size_t bytes = 0;
};

Cache& cache() {
  static Cache instance;
  return instance;
}

// File identity is checked on every lookup. Reading/parsing/probing is done
// only on a miss. A changed or replaced XML file cannot reuse the old entry.
bool localKey(const ncclPeerInfo* peer, const char* path, bool explicitFile, std::string* key) {
  *key = "local";
  append(key, snapshotVersion);
  appendString(key, path);
  append(key, explicitFile);
  append(key, peer->busId);
  append(key, peer->gpuUuid);
  append(key, peer->mloPart);
  uint64_t host = getHostHash();
  append(key, host);
  struct stat st;
  int found = stat(path, &st) == 0;
  append(key, found);
  if (!found) return errno == ENOENT; // Retry other failures via the original reader.
  append(key, st.st_dev);
  append(key, st.st_ino);
  append(key, st.st_size);
#if defined(NCCL_OS_LINUX)
  append(key, st.st_mtim.tv_sec);
  append(key, st.st_mtim.tv_nsec);
  append(key, st.st_ctim.tv_sec);
  append(key, st.st_ctim.tv_nsec);
#else
  append(key, st.st_mtime);
  append(key, st.st_ctime);
#endif
  return true;
}
} // namespace

int ncclTopoCacheMode() {
  return std::max(0, std::min(2, int(ncclParamTopoCache())));
}

ncclResult_t ncclTopoPackXml(const ncclXml* xml, std::string* packed) {
  if (xml->maxIndex <= 0) return ncclInternalError;
  packed->clear();
  append(packed, snapshotVersion);
  uint32_t count = 0;
  append(packed, count);
  packNode(xml->nodes, packed, &count);
  memcpy(&(*packed)[sizeof(snapshotVersion)], &count, sizeof(count));
  return ncclSuccess;
}

ncclResult_t ncclTopoUnpackXml(const std::string& packed, int minNodes, ncclXml** xml) {
  *xml = nullptr;
  Reader in{packed};
  uint32_t version, count;
  if (!in.get(&version) || version != snapshotVersion || !in.get(&count) || count == 0 || count > packed.size())
    return ncclInternalError;
  NCCLCHECK(xmlAlloc(xml, std::max(minNodes, int(count))));
  ncclResult_t ret = unpackNode(&in, *xml, nullptr);
  if (ret == ncclSuccess && (in.pos != packed.size() || (*xml)->maxIndex != int(count))) ret = ncclInternalError;
  if (ret != ncclSuccess) {
    free(*xml);
    *xml = nullptr;
  }
  return ret;
}

ncclTopoSnapshot ncclTopoCacheFind(const std::string& key) {
  Cache& c = cache();
  std::lock_guard<std::mutex> lock(c.mutex);
  auto it = c.index.find(key);
  if (it == c.index.end()) return {};
  c.entries.splice(c.entries.begin(), c.entries, it->second);
  return it->second->data;
}

void ncclTopoCacheInsert(const std::string& key, const std::string& packed) {
  size_t bytes = sizeof(Entry) + 2 * key.size() + packed.size();
  int64_t limit = ncclParamTopoCacheMaxBytes();
  if (limit <= 0 || bytes > size_t(limit)) return;
  Cache& c = cache();
  std::lock_guard<std::mutex> lock(c.mutex);
  if (c.index.find(key) != c.index.end()) return;
  while (c.bytes + bytes > size_t(limit)) {
    const Entry& last = c.entries.back();
    c.bytes -= last.bytes;
    c.index.erase(last.key);
    c.entries.pop_back();
  }
  c.entries.push_front({key, std::make_shared<const std::string>(packed), bytes});
  c.index.emplace(key, c.entries.begin());
  c.bytes += bytes;
}

void ncclTopoCacheClear() {
  Cache& c = cache();
  std::lock_guard<std::mutex> lock(c.mutex);
  c.index.clear();
  c.entries.clear();
  c.bytes = 0;
}

ncclResult_t ncclTopoProbeLocal(const ncclPeerInfo* peer, ncclTopoLocalProbe probe, ncclXml** xml) {
  *xml = nullptr;
  const char* explicitPath = ncclGetEnv("NCCL_TOPO_FILE");
  const char* path = explicitPath ? explicitPath : "/var/run/nvidia-topologyd/virtualTopology.xml";
  std::string key;
  bool enabled = ncclTopoCacheMode() >= 1 && localKey(peer, path, explicitPath != nullptr, &key);
  // Serialize first probes only, never bootstrap or per-comm network work.
  static std::mutex probeMutex;
  std::unique_lock<std::mutex> lock(probeMutex, std::defer_lock);
  if (enabled) {
    lock.lock();
    auto saved = ncclTopoCacheFind(key);
    if (saved) {
      NCCLCHECK(ncclTopoUnpackXml(*saved, NCCL_TOPO_XML_MAX_NODES, xml));
      INFO(NCCL_GRAPH, "TOPO/CACHE local hit");
      return ncclSuccess;
    }
  }
  NCCLCHECK(xmlAlloc(xml, NCCL_TOPO_XML_MAX_NODES));
  ncclResult_t ret = probe(peer->busId, explicitPath, *xml);
  if (ret != ncclSuccess) {
    free(*xml);
    *xml = nullptr;
    return ret;
  }
  if (enabled) {
    std::string after, packed;
    // Do not publish a probe if the file was replaced while it was being read.
    if (localKey(peer, path, explicitPath != nullptr, &after) && key == after) {
      ret = ncclTopoPackXml(*xml, &packed);
      if (ret == ncclSuccess) ncclTopoCacheInsert(key, packed);
    }
    INFO(NCCL_GRAPH, "TOPO/CACHE local probe");
  }
  return ncclSuccess;
}

namespace {
struct ContributionInfo {
  uint64_t digest[2];
  int eligible;
};

void activeGpus(ncclXmlNode* node, std::vector<ncclXmlNode*>* gpus) {
  if (strcmp(node->name, "gpu") == 0) gpus->push_back(node);
  for (int s = 0; s < node->nSubs; s++) activeGpus(node->subs[s], gpus);
}

// Strip only this contribution's comm rank. Keep effective GDR, network view,
// XML overrides and child order in the signature. Unusual multi-GPU injected
// contributions take the original exchange/fusion path.
ncclResult_t packContribution(ncclXml* xml, int rank, std::string* packed, bool* eligible) {
  *eligible = false;
  std::vector<ncclXmlNode*> gpus;
  activeGpus(xml->nodes, &gpus);
  ncclXmlNode* owner = nullptr;
  for (auto* gpu : gpus) {
    int r;
    NCCLCHECK(xmlGetAttrIntDefault(gpu, "rank", &r, -1));
    if (r < 0) continue;
    if (r != rank || owner != nullptr) return ncclSuccess;
    owner = gpu;
  }
  if (owner == nullptr) return ncclSuccess;
  NCCLCHECK(xmlSetAttrInt(owner, "rank", 0));
  ncclResult_t ret = ncclTopoPackXml(xml, packed);
  NCCLCHECK(xmlSetAttrInt(owner, "rank", rank));
  *eligible = ret == ncclSuccess;
  return ret;
}

ContributionInfo contributionInfo(const ncclPeerInfo& peer, const std::string& packed, bool eligible) {
  ContributionInfo info{};
  info.eligible = eligible;
  uint64_t a[2] = {1, 1}, b[2] = {snapshotVersion, 0x6e63636c746f706f};
  eatHash(a, &peer.gpuUuid);
  eatHash(a, packed.data(), packed.size());
  eatHash(b, &peer.gpuUuid);
  eatHash(b, packed.data(), packed.size());
  info.digest[0] = digestHash(a);
  info.digest[1] = digestHash(b);
  return info;
}

std::string fragmentKey(const ContributionInfo& info) {
  std::string key = "fragment";
  key.append(reinterpret_cast<const char*>(info.digest), sizeof(info.digest));
  return key;
}

std::string systemKey(const std::vector<ContributionInfo>& info) {
  std::string key = "system";
  append(&key, snapshotVersion);
  // Ordered physical contributions, not communicator IDs or old comm ranks.
  for (auto& member : info) key.append(reinterpret_cast<const char*>(member.digest), sizeof(member.digest));
  return key;
}

ncclResult_t mapRanks(ncclXml* xml, const std::vector<int>& rankMap) {
  for (int i = 0; i < xml->maxIndex; i++) {
    ncclXmlNode* node = xml->nodes + i;
    if (strcmp(node->name, "gpu") != 0) continue;
    int rank;
    NCCLCHECK(xmlGetAttrIntDefault(node, "rank", &rank, -1));
    if (rank < 0) continue;
    if (rank >= int(rankMap.size()) || rankMap[rank] < 0) return ncclInternalError;
    NCCLCHECK(xmlSetAttrInt(node, "rank", rankMap[rank]));
  }
  return ncclSuccess;
}

using OwnedXml = std::unique_ptr<ncclXml, decltype(&free)>;

ncclResult_t fuseContributions(const std::vector<ncclTopoSnapshot>& fragments, ncclXml** xmlOut) {
  std::vector<OwnedXml> parts;
  int maxNodes = 0;
  for (auto& fragment : fragments) {
    ncclXml* xml;
    NCCLCHECK(ncclTopoUnpackXml(*fragment, 0, &xml));
    parts.emplace_back(xml, free);
    maxNodes += xml->maxIndex;
  }
  ncclXml* xml;
  NCCLCHECK(xmlAlloc(&xml, maxNodes));
  OwnedXml fused(xml, free);
  for (int i = 0; i < int(parts.size()); i++) {
    NCCLCHECK(mapRanks(parts[i].get(), std::vector<int>{i}));
    NCCLCHECK(ncclTopoFuseXml(fused.get(), parts[i].get()));
  }
  *xmlOut = fused.release();
  return ncclSuccess;
}
} // namespace

static ncclResult_t exchangeNodeXml(struct ncclComm* comm, std::vector<int>& localRanks, int localRank,
                                    ncclXml** xmlOut, std::vector<ncclTopoSnapshot>* contributions) {
  ncclResult_t ret = ncclSuccess;
  ncclXml* xml = *xmlOut;
  char* mem = nullptr;
  ncclXml* rankXml;
  int nLocalRanks = localRanks.size();
  int fusedMaxNodes = 0;
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  rankXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  INFO(NCCL_GRAPH, "TOPO/CACHE node XML exchange ranks %d", nLocalRanks);
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks.data(), localRank, nLocalRanks, mem,
                                            xmlMemSize(NCCL_TOPO_XML_MAX_NODES)),
                ret, fail);
  // ncclTopoFuseXml only copies existing source nodes into dst (never adds new
  // ones), so fusedMaxNodes is a true upper bound on the fused node count.
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * i);
    fusedMaxNodes += peerXml->maxIndex;
  }
  if (fusedMaxNodes > xml->maxNodes) {
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, fusedMaxNodes), ret, fail);
  } else {
    // The original topology is preserved in mem, so reuse this allocation as the empty fusion destination.
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), ret, fail);
    if (contributions) {
      std::string packed;
      bool eligible;
      NCCLCHECKGOTO(packContribution(peerXml, localRanks[i], &packed, &eligible), ret, fail);
      if (eligible) (*contributions)[i] = std::make_shared<const std::string>(std::move(packed));
    }
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

exit:
  *xmlOut = xml;
  free(mem);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTopoGetNodeSnapshot(struct ncclComm* comm, ncclXml** xmlOut) {
  std::vector<int> localRanks;
  int localRank = -1;
  if (comm->MNNVL) {
    localRanks.assign(comm->clique.ranks, comm->clique.ranks + comm->clique.size);
    return exchangeNodeXml(comm, localRanks, comm->cliqueRank, xmlOut, nullptr);
  }
  int mode = 2;
  for (int r = 0; r < comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash) {
      if (r == comm->rank) localRank = localRanks.size();
      localRanks.push_back(r);
      mode = std::min(mode, comm->peerInfo[r].topoCacheMode);
    }
  }
  // Negotiated through the existing peer-info AllGather. Even a rank with
  // caching disabled follows the same sequence of collectives as its peers.
  if (mode < 2) return exchangeNodeXml(comm, localRanks, localRank, xmlOut, nullptr);

  int nLocalRanks = localRanks.size();
  std::string packed;
  bool eligible;
  NCCLCHECK(packContribution(*xmlOut, comm->rank, &packed, &eligible));
  std::vector<ContributionInfo> info(nLocalRanks);
  info[localRank] = contributionInfo(comm->peerInfo[comm->rank], packed, eligible);
  // Revalidate every contributor's current XML/plugin view with a small record.
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks.data(), localRank, nLocalRanks,
                                        info.data(), sizeof(ContributionInfo)));
  for (auto& member : info) {
    if (!member.eligible) return exchangeNodeXml(comm, localRanks, localRank, xmlOut, nullptr);
  }

  const std::string key = systemKey(info);
  auto saved = ncclTopoCacheFind(key);
  std::vector<ncclTopoSnapshot> fragments(nLocalRanks);
  std::vector<int> hits(nLocalRanks, 0);
  hits[localRank] = saved != nullptr;
  if (!saved) {
    hits[localRank] = 1;
    for (int i = 0; i < nLocalRanks; i++) {
      fragments[i] = i == localRank ? std::make_shared<const std::string>(packed) :
                                      ncclTopoCacheFind(fragmentKey(info[i]));
      if (!fragments[i]) hits[localRank] = 0;
    }
  }
  // No registry/probe/plugin lock is held across bootstrap. Hold shared_ptrs
  // through agreement so concurrent eviction cannot invalidate a promised hit.
  NCCLCHECK(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks.data(), localRank, nLocalRanks,
                                        hits.data(), sizeof(int)));
  bool allHit = std::all_of(hits.begin(), hits.end(), [](int hit) { return hit != 0; });
  if (!allHit) {
    // One miss sends every participant down the original XML exchange path.
    INFO(NCCL_GRAPH, "TOPO/CACHE node miss ready %d/%d", int(std::count(hits.begin(), hits.end(), 1)), nLocalRanks);
    NCCLCHECK(exchangeNodeXml(comm, localRanks, localRank, xmlOut, &fragments));
    for (int i = 0; i < nLocalRanks; i++) {
      if (!fragments[i]) return ncclSuccess;
      auto actual = contributionInfo(comm->peerInfo[localRanks[i]], *fragments[i], true);
      if (memcmp(actual.digest, info[i].digest, sizeof(actual.digest)) != 0) return ncclSuccess;
    }
    for (int i = 0; i < nLocalRanks; i++) ncclTopoCacheInsert(fragmentKey(info[i]), *fragments[i]);
    // Store local slot numbers. The new comm's rank map is applied to a private
    // copy on lookup, including non-contiguous DP ranks and reordered groups.
    std::vector<int> toLocal(comm->nRanks, -1);
    for (int i = 0; i < nLocalRanks; i++) toLocal[localRanks[i]] = i;
    NCCLCHECK(mapRanks(*xmlOut, toLocal));
    ncclResult_t ret = ncclTopoPackXml(*xmlOut, &packed);
    NCCLCHECK(mapRanks(*xmlOut, localRanks));
    NCCLCHECK(ret);
    ncclTopoCacheInsert(key, packed);
    return ncclSuccess;
  }

  ncclXml* xml = nullptr;
  if (saved) {
    NCCLCHECK(ncclTopoUnpackXml(*saved, 0, &xml));
  } else {
    // A warmed larger group supplies the selected members' immutable facts.
    // Fuse only those contributions, in the new group's order. This preserves
    // the legacy NIC union and GPU order instead of pruning a foreign graph.
    NCCLCHECK(fuseContributions(fragments, &xml));
  }
  OwnedXml result(xml, free);
  if (!saved) {
    NCCLCHECK(ncclTopoPackXml(xml, &packed));
    ncclTopoCacheInsert(key, packed);
  }
  NCCLCHECK(mapRanks(xml, localRanks));
  free(*xmlOut);
  *xmlOut = result.release();
  INFO(NCCL_GRAPH, "TOPO/CACHE node reuse ranks %d (%s)", nLocalRanks, saved ? "snapshot" : "projection");
  return ncclSuccess;
}
