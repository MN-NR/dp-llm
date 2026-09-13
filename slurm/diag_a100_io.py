#!/usr/bin/env python
"""One-off diagnostic: why did Llama-3.2-1B-Instruct's from_pretrained() hang
for 10+ minutes on Ascend (two different nodes) when it took ~2s on Cardinal?
Isolates raw filesystem throughput from the HF/safetensors loading path,
single-process (no DDP, no NCCL) so a stall shows up directly instead of as
a downstream NCCL timeout.
"""
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_HOME = os.environ.get("HF_HOME", "")

print(f"HF_HOME = {HF_HOME}")
snap_root = os.path.join(HF_HOME, "hub", "models--meta-llama--Llama-3.2-1B-Instruct", "snapshots")
snap = None
if os.path.isdir(snap_root):
    subs = os.listdir(snap_root)
    if subs:
        snap = os.path.join(snap_root, subs[0])
print(f"snapshot dir = {snap}")

if snap:
    weights = os.path.join(snap, "model.safetensors")
    size = os.path.getsize(weights)
    print(f"weights file: {weights}  ({size/1e9:.2f} GB)")

    t0 = time.time()
    with open(weights, "rb") as f:
        n = 0
        chunk = 1 << 20
        while True:
            b = f.read(chunk)
            if not b:
                break
            n += len(b)
    t1 = time.time()
    print(f"[raw sequential read] {n/1e9:.2f} GB in {t1-t0:.1f}s -> {n/1e9/max(t1-t0,1e-6):.2f} GB/s")

print()
print("[torch/transformers load, single GPU, no DDP]")
t0 = time.time()
import torch
from transformers import AutoModelForCausalLM
t1 = time.time()
print(f"  import torch+transformers: {t1-t0:.1f}s")

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    token=os.environ.get("HF_TOKEN"),
)
t1 = time.time()
print(f"  from_pretrained (CPU):     {t1-t0:.1f}s")

t0 = time.time()
model = model.to("cuda:0")
torch.cuda.synchronize()
t1 = time.time()
print(f"  .to('cuda:0'):             {t1-t0:.1f}s")

print("\nDONE -- no hang.")
