#!/usr/bin/env python3
"""Exercise cache levels, partial misses and asymmetric configuration."""
import os
import pathlib
import subprocess
import sys
import tempfile
import time

binary = str(pathlib.Path(sys.argv[1]).resolve())
mode = sys.argv[2] if len(sys.argv) > 2 else "2"
nranks = int(sys.argv[3]) if len(sys.argv) > 3 else 8
assert mode in ("0", "1", "2", "mixed", "tiny")
directory = pathlib.Path(tempfile.mkdtemp(prefix=f"nccl-topology-{mode}-"))
processes, logs = [], []
environment = {k: v for k, v in os.environ.items() if not k.startswith("NCCL_")}
try:
    for rank in range(nranks):
        env = {**environment, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
               "NCCL_TOPO_CACHE": "0" if mode == "mixed" and rank == 0 else
                                  "2" if mode in ("mixed", "tiny") else mode}
        if mode == "tiny":
            env["NCCL_TOPO_CACHE_MAX_BYTES"] = "1"
        log = (directory / f"rank-{rank}.log").open("w")
        logs.append(log)
        processes.append(subprocess.Popen([binary, str(rank), str(nranks), str(directory)],
                                          stdout=log, stderr=subprocess.STDOUT, env=env))
    deadline = time.monotonic() + 300
    for process in processes:
        if process.wait(timeout=max(1, deadline - time.monotonic())) != 0:
            raise RuntimeError("rank failed")
    for rank in range(nranks):
        text = (directory / f"rank-{rank}.log").read_text()
        assert "PASS rank" in text
        assert text.count("PASS exact XML equivalence") == 6
        phases = {part.splitlines()[0]: part for part in text.split("PHASE ")[1:]}
        if mode == "2":
            assert "node XML exchange" in phases["cold"]
            assert "node reuse ranks" in phases["independent"] and "(snapshot)" in phases["independent"]
            for phase in ("reordered", "tp", "dp"):
                assert "node reuse ranks" in phases[phase], (rank, phase)
                assert "node XML exchange" not in phases[phase], (rank, phase)
            assert "node XML exchange" in phases["partial-miss"]
            assert f"node miss ready {nranks - 1}/{nranks}" in phases["partial-miss"]
            assert "node reuse ranks" in phases["strided-local"]
            assert "node reuse ranks" in phases["recreated"]
        else:
            assert "node XML exchange" in phases["independent"]
            assert "node reuse ranks" not in phases["independent"]
        if mode == "1":
            assert "local hit" in phases["independent"]
        if mode == "0":
            assert "local hit" not in text
    print(f"PASS mode={mode}, ranks={nranks}: exact XML, rank ordering, cache paths, collectives, concurrent init")
finally:
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()
    for log in logs:
        log.close()
    print(f"Logs: {directory}")
