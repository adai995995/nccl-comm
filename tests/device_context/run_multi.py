#!/usr/bin/env python3
"""Launch independent rank processes and retain their logs on failure."""
import os
import pathlib
import subprocess
import sys
import tempfile

binary = str(pathlib.Path(sys.argv[1]).resolve())
nranks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
directory = pathlib.Path(tempfile.mkdtemp(prefix="nccl-device-context-"))
processes = []
logs = []
try:
    for rank in range(nranks):
        log = (directory / f"rank-{rank}.log").open("w")
        logs.append(log)
        processes.append(subprocess.Popen([binary, str(rank), str(nranks), str(directory)],
                                          stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}))
    for process in processes:
        if process.wait(timeout=180) != 0:
            raise RuntimeError("rank failed")
    for rank in range(nranks):
        text = (directory / f"rank-{rank}.log").read_text()
        assert "PASS rank" in text, text
        assert text.count("DeviceFacts: probed") == 1, text
        assert text.count("ContextEnv: initialized") == 2, text
        assert text.count("ContextEnv: released") == 2, text
    print(f"PASS {nranks} ranks: one device probe and two ContextEnv lifetimes per rank")
finally:
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()
    for log in logs:
        log.close()
    print(f"Logs: {directory}")
