#!/usr/bin/env python
"""Isolates whether the 10-min hang is specifically from TWO processes on
the same node loading the model at once (no NCCL/torchrun involved at all --
just plain subprocesses), with a hard 3-minute cap per process so this job
doesn't need the full 10 minutes to give an answer.
"""
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER = """
import os, sys, time
t0 = time.time()
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN"),
)
model = model.to(f"cuda:{sys.argv[1]}")
torch.cuda.synchronize()
print(f"rank{sys.argv[1]} OK in {time.time()-t0:.1f}s", flush=True)
"""

open("/tmp/_diag_worker.py", "w").write(WORKER)

print("Launching 2 concurrent from_pretrained processes (one per GPU)...", flush=True)
t0 = time.time()
procs = []
for gpu in (0, 1):
    env = dict(os.environ)
    p = subprocess.Popen(
        [sys.executable, "/tmp/_diag_worker.py", str(gpu)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    procs.append(p)

CAP = 180  # seconds
deadline = t0 + CAP
results = {}
while time.time() < deadline and len(results) < 2:
    for i, p in enumerate(procs):
        if i in results:
            continue
        if p.poll() is not None:
            out = p.stdout.read()
            results[i] = (p.returncode, out, time.time() - t0)
    time.sleep(1)

for i, p in enumerate(procs):
    if i not in results:
        print(f"proc {i}: STILL RUNNING after {CAP}s -> HANG CONFIRMED (killing)")
        p.kill()
    else:
        rc, out, dt = results[i]
        print(f"proc {i}: finished in {dt:.1f}s rc={rc}\n{out}")

print(f"\ntotal wall time: {time.time()-t0:.1f}s")
