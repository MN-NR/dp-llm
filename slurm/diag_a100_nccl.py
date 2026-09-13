#!/usr/bin/env python
"""Minimal reproduction: does a bare NCCL init + barier() between 2 GPUs on
one Ascend node hang? This is the FIRST collective op train_dp_ddp.py does
(right after dist.init_process_group), before any model/data loading -- if
this hangs, the earlier "stuck in from_pretrained" stack traces were a
background thread's frame at abort time, not the real blocking point.
60s init timeout (vs NCCL's 600s default) so this fails fast if it hangs.
"""
import datetime
import os
import time

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

t0 = time.time()
print(f"[rank {local_rank}] init_process_group...", flush=True)
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=60))
print(f"[rank {local_rank}] init_process_group OK in {time.time()-t0:.1f}s, world_size={dist.get_world_size()}", flush=True)

t0 = time.time()
print(f"[rank {local_rank}] barrier()...", flush=True)
dist.barrier()
print(f"[rank {local_rank}] barrier() OK in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
x = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(x)
torch.cuda.synchronize()
print(f"[rank {local_rank}] all_reduce OK in {time.time()-t0:.1f}s, result={x.item()}", flush=True)

dist.destroy_process_group()
print(f"[rank {local_rank}] DONE, no hang.", flush=True)
