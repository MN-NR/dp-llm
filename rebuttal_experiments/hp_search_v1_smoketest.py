"""One pilot=True run (num_epochs forced to 1, full dataset for DP accounting)
as a fast correctness gate before committing to the full ~2-day, 9-run Stage 1
sweep in hp_search_v1_stage1.py. Uses the base_lr/3-epoch config's settings
(the num_epochs=3 is irrelevant here since pilot=True overrides it to 1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hp_search_v1_stage1 import build_cfg, EPSILON, SEED, BASE_LR
from finetune.train_dp import train_dp

if __name__ == "__main__":
    cfg, tag = build_cfg(BASE_LR, 3)
    print(f"Smoke test (pilot=True) for {tag}", flush=True)
    train_dp(cfg, epsilon=EPSILON, pilot=True, seed=SEED)
