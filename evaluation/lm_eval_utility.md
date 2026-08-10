# lm-evaluation-harness (utility eval)

This repo saves LoRA/PEFT checkpoints under `checkpoints/`. You can run zero-shot utility evaluation on those saved checkpoints using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) without using this repo's training datasets.

## Install

In your environment:

```bash
pip install -U lm-eval
```

Expose the repo wrapper scripts:

```bash
export PATH="$PWD/bin:$PATH"
```

## Run on a single checkpoint

Matches the command you requested:

```bash
bashlm_eval --model hf \
  --model_args pretrained=<base>,peft=<checkpoint>,dtype=bfloat16 \
  --tasks hellaswag,arc_easy,boolq,piqa \
  --device cuda --batch_size 4
```

Example using a saved checkpoint in this repo:

```bash
bashlm_eval --model hf \
  --model_args pretrained=google/gemma-2b-it,peft=checkpoints/gemma_2b_it/baseline/final,dtype=bfloat16 \
  --tasks hellaswag,arc_easy,boolq,piqa \
  --device cuda --batch_size 4
```

## Convenience runner

```bash
bin/run_lm_eval_utility.sh --base <hf_model_id_or_path> --peft <checkpoint_dir>
```

Example:

```bash
bin/run_lm_eval_utility.sh \
  --base google/gemma-2b-it \
  --peft checkpoints/gemma_2b_it/baseline/final
```

## VaultGemma DP sweep (eps 3/5/8)

Runs utility eval for:
- `checkpoints/vault_gemma/dp_eps3.0/{final|best}`
- `checkpoints/vault_gemma/dp_eps5.0/{final|best}`
- `checkpoints/vault_gemma/dp_eps8.0/{final|best}`

```bash
bin/run_lm_eval_vault_gemma_dp.sh --which final
```
