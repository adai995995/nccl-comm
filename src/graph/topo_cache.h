/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TOPO_CACHE_H_
#define NCCL_TOPO_CACHE_H_

#include "xml.h"
#include <memory>
#include <string>

struct ncclComm;
struct ncclPeerInfo;

// Process-local, immutable CPU snapshots. No CUDA context, communicator or
// network-plugin handles are retained. Every caller receives a private XML.
int ncclTopoCacheMode();
bool ncclTopoNodeCacheWarm();
using ncclTopoLocalProbe = ncclResult_t (*)(int64_t busId, const char* topoFile, ncclXml* xml);
ncclResult_t ncclTopoProbeLocal(const ncclPeerInfo* peer, ncclTopoLocalProbe probe, ncclXml** xml);

// Compact representation of the reachable XML tree, without addresses or
// unused fixed-capacity nodes. Attribute and child order are preserved.
ncclResult_t ncclTopoPackXml(const ncclXml* xml, std::string* packed);
ncclResult_t ncclTopoUnpackXml(const std::string& packed, int minNodes, ncclXml** xml);

using ncclTopoSnapshot = std::shared_ptr<const std::string>;
ncclTopoSnapshot ncclTopoCacheFind(const std::string& key);
void ncclTopoCacheInsert(const std::string& key, const std::string& packed);
// Internal invalidation point; live readers retain their immutable snapshot.
void ncclTopoCacheClear();

ncclResult_t ncclTopoGetNodeSnapshot(struct ncclComm* comm, ncclXml** xml);
// Caller owns the assembled, communicator-specific XML (also useful to compare
// the cached and uncached paths before graph conversion/search).
ncclResult_t ncclTopoGetCommXml(struct ncclComm* comm, const char* dumpXmlFile, ncclXml** xml);

#endif
