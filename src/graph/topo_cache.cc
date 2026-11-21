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

NCCL_PARAM(TopoCache, "TOPO_CACHE", 1);
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
  for (int a = 0; a < nAttrs; a++) {
    char key[MAX_STR_LEN + 1], value[MAX_STR_LEN + 1];
    if (!in->string(key) || !in->string(value)) return ncclInternalError;
    NCCLCHECK(xmlSetAttr(node, key, value));
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

ncclResult_t ncclTopoGetNodeSnapshot(struct ncclComm* comm, ncclXml** xmlOut) {
  ncclResult_t ret = ncclSuccess;
  ncclXml* xml = *xmlOut;
  char* mem = nullptr;
  int* localRanks = nullptr;
  ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  int fusedMaxNodes = 0;
  // XML topo fusion.
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), ret, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank) localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), ret, fail);
  rankXml = (struct ncclXml*)(mem + xmlMemSize(NCCL_TOPO_XML_MAX_NODES) * localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), ret, fail);
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem,
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
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), ret, fail);
  }

exit:
  *xmlOut = xml;
  if (!comm->MNNVL) free(localRanks);
  free(mem);
  return ret;
fail:
  goto exit;
}
