# OSC Cardinal — Llama-3.2-1B-Instruct DP run (ε = 5.0)

Everything here runs on **OSC Cardinal**. You edit code + `git` + `sbatch` on
a `cardinal-loginNN` node; the actual GPU work runs on a Cardinal compute
node with 2× H100 via Slurm. Never run training on the login node.

```
Mac / VS Code
   │  SSH
   ▼
cardinal-loginNN     ← edit code, git, sbatch, dataset build
   │  Slurm
   ▼
Cardinal compute node (2× H100, partition `gpu`)
   │
   ▼
finetune/train_dp_ddp.py   (torchrun --nproc_per_node=2, Opacus DPDDP)
```

Account: **`pmiu0248`**.  GPU nodes: `c0801–c0825`, 4× H100 each, 104 cores,
~936 GB RAM.  All GPU jobs here use the **`gpu`** partition (`cardinal-gpu-
partition` QOS: up to 4 GPU/job, 7-day limit). The `debug` partition is
**not usable** — its `cardinal-debug` QOS caps jobs at 1 GPU/node
(`QOSMaxGRESPerNode`) and every job here needs 2. Keep `--time` short on the
smoke/benchmark jobs so they still schedule quickly.

---

## One-time setup

### 1. Python environment

```bash
cd ~/dp-llm
bash bin/osc_setup_env.sh
```

Builds `.venv/` (git-ignored): Python 3.11, **torch 2.5.1+cu124**,
`transformers==4.57.6`, `opacus==1.5.4`, `peft==0.18.1` + eval/plot deps.
torch 2.5.x matches the frozen LS04 pipeline; the cu124 wheel (vs LS04's
cu121) is the deliberate choice for H100 on Cardinal's CUDA-13 driver — its
bundled CUDA runtime + NCCL mean **no `module load cuda` is needed**. Full
transitive lock: `slurm/requirements.lock.txt`.

### 2. Hugging Face token (gated model)

`meta-llama/Llama-3.2-1B-Instruct` is gated. Accept the license on HF, then:

```bash
mkdir -p ~/.config/dp-llm
printf '%s' 'hf_xxxxxxxxxxxxxxxxxxxx' > ~/.config/dp-llm/hf_token
chmod 600 ~/.config/dp-llm/hf_token
```

`slurm/env.sh` reads that file and exports `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN`. It is outside the repo and never committed.

### 3. Build the processed dataset (seed 42)

```bash
sbatch slurm/build_dataset.slurm      # cpu partition, ~10–30 min
```

Rebuilds `instruction_recall_dataset/dataset/processed/` deterministically
(seed 42) — the exact path `finetune/config.py`'s `llama3_2_1b_it` entry
expects — and verifies every required file exists. Pulls `ag_news`,
`glue/mnli`, `ai4privacy/open-pii-masking-500k` from the HF Hub (confirmed:
Cardinal compute nodes have outbound internet).

HF cache lives on scratch: `HF_HOME=/fs/scratch/PMIU0222/hf_cache`.
Checkpoints + results stay in-repo (home), git-ignored.

---

## Pre-run checklist (do these before the paper run)

### Step 1 + 2 — smoke test: tokenizer, EOS formatting, 50 DPDDP steps

```bash
sbatch slurm/train_test.slurm         # gpu partition, short --time, ~15–30 min
```

Runs `slurm/verify_llama_tokenizer.py` (chat template, EOS token, and that
every training example ends in a **supervised** EOS via the canonical
`canarybench.prompt_format` formatter + the real `MultiTaskDataset` path),
then 50 real DP-SGD steps at the paper config. No valid training output —
just proves the pipeline, accountant, NCCL and checkpoint dir all work.

### Step 3 — physical-batch benchmark (global batch fixed at 128)

```bash
sbatch slurm/bench_physical_batch.slurm   # gpu partition, short --time, ~30–45 min
```

Runs `--physical-batch-size 32` then `64` at global logical batch 128
(`--batch-size 16 --grad-accum 8`), 30 optimizer steps each, sampling
`nvidia-smi`. Prints **sec/optimizer-step, mean+peak GPU util, peak VRAM,
non-finite step count** per config. Pick the largest physical batch that
didn't OOM, had 0 non-finite steps, and has the lowest sec/step.

---

## The paper run

Edit `slurm/train_paper.slurm`: set `PHYS_BATCH=` to the value chosen from
the benchmark. Then (after the user says go):

```bash
sbatch slurm/train_paper.slurm        # gpu partition, --time 24:00:00
```

Config (frozen corrected pipeline): `llama3_2_1b_it`, ε 5.0, δ 1e-5,
`max_grad_norm 1.0` (from `config.py`), seed 42, global batch **128**
(`bs 16 × ga 8`), LR **2e-5**, **20 epochs**, `--save-every-epoch`, LoRA
target modules from `config.py`, same Opacus accountant. Test set
(`combined_test.jsonl`) is never read. Per-epoch adapters land under
`checkpoints/instruction_recall/llama3_2_1b_it/dp_eps5.0_seed42_batch128_epochs20_2gpu/`.

Do **not** run Gemma-3 ε=5 — LS04 already did it.

---

## After training

### Checkpoint selection — validation utility ONLY

Score every saved epoch adapter on **validation** with:
AG News greedy-gen accuracy, MNLI greedy-gen accuracy, PII Unique-F1, PII
exact-match, repetition rate, invalid-output rate. **Do not** use any
memorization/exposure metric to choose the checkpoint.

`benchmarks/run_profile_matrix.py` (→ `benchmarks.cross_model_exposure.
evaluate_checkpoint`) computes the utility side per adapter against
`instruction_recall_dataset/dataset/processed/combined_val.jsonl`. Point it
at each epoch's adapter dir (confirm how `_iter_runs` discovers them, or
call `evaluate_checkpoint(base_model_path=..., adapter_path=<epoch adapter>,
val_path=<...combined_val.jsonl>)` directly per epoch) and take the argmax
on the utility composite.

### Privacy audit — on the selected checkpoint only

Report: actual ε (from the run's results JSON / accountant), member vs
reference losses, loss **gap + bootstrap CI**, **WBC AUC + CI**, exposure,
the **1× / 10× / 50×** repetition breakdown, and targeted greedy
extraction. Path: `benchmarks/run_profile_matrix.py --models llama3_2_1b_it
--eps 5` for the selected adapter → `bin/privacy_report.py --profile
<json>` and `bin/exposure_tables.py --glob '<profile_matrix>/*.json'`
for the gap/CI/AUC/exposure tables; `benchmarks/repetition_study.py` for
the 1×/10×/50× split.

**Then stop and report before launching the next configuration.**

---

## Interactive debugging (optional)

```bash
salloc --account=pmiu0248 --partition=gpu --nodes=1 --gpus-per-node=2 \
       --cpus-per-task=16 --mem=128G --time=01:00:00
srun --pty bash
hostname && nvidia-smi          # should be a c08xx node with 2× H100
cd ~/dp-llm && source slurm/env.sh
```

## Files

| file | what |
|---|---|
| `bin/osc_setup_env.sh` | build `.venv/` (torch 2.5.1+cu124 + pinned deps) |
| `slurm/env.sh` | sourced by every job: venv, `HF_HOME`, HF token, determinism |
| `slurm/build_dataset.slurm` | rebuild processed dataset, seed 42 (cpu) |
| `slurm/verify_llama_tokenizer.py` | tokenizer + EOS-supervised formatting check (CPU) |
| `slurm/train_test.slurm` | verify + 50-step DPDDP smoke (gpu partition, 2× H100) |
| `slurm/bench_physical_batch.slurm` | physical batch 32 vs 64 @ global 128 (gpu partition) |
| `slurm/train_paper.slurm` | the 20-epoch ε=5 run (gpu, 2× H100) — set `PHYS_BATCH` first |
| `slurm/requirements.lock.txt` | full transitive lock of `.venv/` |
