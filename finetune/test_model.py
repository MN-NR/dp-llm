#python finetune/test_model.py --model gemma_2b_it --checkpoint checkpoints/gemma_2b_it/dp_eps16.0/final --dataset all --data-dir instruction_recall_dataset/dataset/processed --device cuda
import os, sys, json, argparse, re
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from finetune.config import MODELS, repo_path
from finetune.dataset import MultiTaskDataset
from evaluation.eval_metrics import compute_all_metrics, print_metrics_table
from evaluation.loss_based_eval import evaluate_classification_by_loss, TASK_LABELS


DATASET_FILES = {
    "test": "combined_test.jsonl",
    "val": "combined_val.jsonl",
    "attack_eval": "attack_eval.jsonl",
}


GEN_TASKS = {
    "qa",
    "summarization",
    "pii_mask",
    "pii_ner",
    "span_corrupt",
    "pii_detection",
}


TASK_MAX_TOKENS = {
    "qa": 30,
    "summarization": 40,
    "pii_mask": 150,
    "pii_ner": 80,
    "span_corrupt": 80,
    "pii_detection": 80,
}


def _norm_ws(x):
    return re.sub(r"\s+", " ", (x or "").lower().strip())


def _split_pii_relaxed(text):
    return [_norm_ws(x) for x in re.split(r"[;\n]", str(text or "")) if x.strip()]


def _relaxed_match(pred, gold, overlap_threshold=0.5):
    pred = _norm_ws(pred)
    gold = _norm_ws(gold)

    if pred == gold:
        return True

    pred_tokens = set(pred.split())
    gold_tokens = set(gold.split())

    if not pred_tokens or not gold_tokens:
        return False

    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens)
    return overlap >= overlap_threshold


def _match_pii(preds, golds, overlap_threshold):
    """Greedily match predicted PII strings against gold strings (each gold
    usable at most once). Matching rule itself (_relaxed_match) is untouched
    -- this is just the shared bipartite-matching loop used by both the
    legacy and unique scoring variants below."""
    matched_gold = set()
    matched_pred = set()
    for pi, p in enumerate(preds):
        for gi, g in enumerate(golds):
            if gi in matched_gold:
                continue
            if _relaxed_match(p, g, overlap_threshold=overlap_threshold):
                matched_gold.add(gi)
                matched_pred.add(pi)
                break
    return matched_gold, matched_pred


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def relaxed_pii_scores(samples, overlap_threshold=0.5):
    """Two PII-detection scoring variants, both using the same matching rule
    (_relaxed_match, same overlap_threshold):

      relaxed_* / pii_legacy_*  -- original behavior, unchanged. Every parsed
        predicted string counts as its own item, duplicates included. A
        model that repeats one correct value N times under greedy decoding
        (a decoding-repetition-loop artifact, not N distinct extra guesses)
        is penalized N-1 times as false positives.

      pii_unique_*  -- normalized predicted values are deduplicated (order
        preserving) before matching. Isolates "did it identify the right
        values" from "did it also loop while decoding." _split_pii_relaxed
        already normalizes each item (via _norm_ws), so dedup is a plain
        set-membership check, no extra normalization needed.

    repetition_* fields describe the raw prediction lists directly, so the
    legacy-vs-unique gap is legible rather than just two numbers that differ.
    """
    tp_legacy = fp_legacy = fn_legacy = exact_legacy = 0
    tp_unique = fp_unique = fn_unique = exact_unique = 0
    examples_with_dupes = 0
    total_dupe_count = 0
    max_dupe_count = 0

    for s in samples:
        golds = _split_pii_relaxed(s.get("target", ""))
        preds = _split_pii_relaxed(s.get("generated", ""))

        matched_gold, matched_pred = _match_pii(preds, golds, overlap_threshold)
        tp_legacy += len(matched_gold)
        fp_legacy += len(preds) - len(matched_pred)
        fn_legacy += len(golds) - len(matched_gold)
        if set(preds) == set(golds):
            exact_legacy += 1

        seen = set()
        unique_preds = []
        for p in preds:
            if p not in seen:
                seen.add(p)
                unique_preds.append(p)
        n_dupes = len(preds) - len(unique_preds)
        if n_dupes > 0:
            examples_with_dupes += 1
            total_dupe_count += n_dupes
            max_dupe_count = max(max_dupe_count, n_dupes)

        matched_gold_u, matched_pred_u = _match_pii(unique_preds, golds, overlap_threshold)
        tp_unique += len(matched_gold_u)
        fp_unique += len(unique_preds) - len(matched_pred_u)
        fn_unique += len(golds) - len(matched_gold_u)
        if set(unique_preds) == set(golds):
            exact_unique += 1

    p_l, r_l, f1_l = _prf(tp_legacy, fp_legacy, fn_legacy)
    p_u, r_u, f1_u = _prf(tp_unique, fp_unique, fn_unique)
    n = len(samples)

    return {
        # Original keys, byte-identical computation -- unchanged for any
        # existing reader.
        "relaxed_precision": round(p_l, 4),
        "relaxed_recall": round(r_l, 4),
        "relaxed_f1": round(f1_l, 4),
        "relaxed_exact_match": round((exact_legacy / n) if n else 0.0, 4),
        # Explicit legacy_* aliases (same values, same computation).
        "pii_legacy_precision": round(p_l, 4),
        "pii_legacy_recall": round(r_l, 4),
        "pii_legacy_f1": round(f1_l, 4),
        # Duplicate-predictions-collapsed variant.
        "pii_unique_precision": round(p_u, 4),
        "pii_unique_recall": round(r_u, 4),
        "pii_unique_f1": round(f1_u, 4),
        "pii_unique_exact_match": round((exact_unique / n) if n else 0.0, 4),
        # Repetition diagnostics.
        "repetition_frac_examples_with_dupes": round((examples_with_dupes / n) if n else 0.0, 4),
        "repetition_mean_dupe_count": round((total_dupe_count / n) if n else 0.0, 4),
        "repetition_max_dupe_count": max_dupe_count,
        "n_samples": n,
    }


def resolve_checkpoint_path(checkpoint_path):
    if not checkpoint_path:
        return None

    p = os.path.expanduser(checkpoint_path.strip())

    if os.path.exists(p):
        return p

    if os.path.isabs(p):
        candidate = os.path.join(project_root, p.lstrip(os.sep))
        if os.path.exists(candidate):
            return candidate

    candidate = os.path.join(project_root, p)
    if os.path.exists(candidate):
        return candidate

    return p


def load_model_and_tokenizer(model_key, checkpoint_path):
    cfg = MODELS[model_key]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=torch.bfloat16,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16,
        )

    resolved = resolve_checkpoint_path(checkpoint_path)
    if resolved and os.path.exists(resolved):
        model = PeftModel.from_pretrained(model, resolved)
        print(f"Loaded LoRA checkpoint: {resolved}")
    else:
        print("Using base model")

    return model, tokenizer, cfg


def get_data_path(dataset_type, data_dir=None):
    if dataset_type not in DATASET_FILES:
        raise ValueError(f"Unknown dataset: {dataset_type}")

    if data_dir:
        return os.path.join(data_dir, DATASET_FILES[dataset_type])

    return repo_path("data_prep", "Dataset", "final", DATASET_FILES[dataset_type])


def evaluate_loss(model, dataset, dataloader, device):
    model.eval()
    model.to(device)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    total_loss = 0.0
    total_count = 0

    group_loss_sum = defaultdict(float)
    group_count = defaultdict(int)

    task_loss_sum = defaultdict(float)
    task_count = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(input_ids.size(0), -1)

            mask = (shift_labels != -100).float()
            per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            start_idx = batch_idx * dataloader.batch_size

            for i in range(input_ids.size(0)):
                rec_idx = start_idx + i
                if rec_idx >= len(dataset.records):
                    break

                rec = dataset.records[rec_idx]
                group = rec.get("group", "unknown")
                task = rec.get("task", "unknown")
                loss = float(per_sample_loss[i].item())

                total_loss += loss
                total_count += 1

                group_loss_sum[group] += loss
                group_count[group] += 1

                task_loss_sum[task] += loss
                task_count[task] += 1

    return {
        "avg_loss": total_loss / total_count if total_count else 0.0,
        "group_losses": {
            g: group_loss_sum[g] / group_count[g]
            for g in group_loss_sum
        },
        "task_losses": {
            t: task_loss_sum[t] / task_count[t]
            for t in task_loss_sum
        },
    }


class RecordsSubset(Dataset):
    def __init__(self, base_dataset, records):
        self.base_dataset = base_dataset
        self.records = records
        # Snapshot pieces needed to encode without mutating base_dataset.
        self.tokenizer = base_dataset.tokenizer
        self.max_length = base_dataset.max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        full_text, prompt_only = self.base_dataset._format_record(rec)

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze().long()
        attention_mask = encoding["attention_mask"].squeeze().long()
        labels = input_ids.clone()

        if prompt_only:
            prompt_len = len(
                self.tokenizer(
                    prompt_only,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                seq_len = int(attention_mask.sum().item())
                start = int(input_ids.numel() - seq_len)
                labels[start : start + prompt_len] = -100
            else:
                labels[:prompt_len] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        if (labels != -100).sum().item() == 0:
            labels[: len(labels) // 2] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _stratified_sample_by_task(records, max_samples, seed=0):
    """
    Deterministic sampling to reduce bias when records are ordered by task/group.
    Samples proportionally by task, then fills remaining slots.
    """
    if max_samples <= 0 or max_samples >= len(records):
        return records

    rng = torch.Generator()
    rng.manual_seed(int(seed))

    by_task = defaultdict(list)
    for r in records:
        by_task[r.get("task", "unknown")].append(r)

    tasks = sorted(by_task.keys())
    sizes = {t: len(by_task[t]) for t in tasks}
    total = sum(sizes.values())

    alloc = {t: (int(max_samples * (sizes[t] / total)) if total else 0) for t in tasks}

    used = sum(alloc.values())
    if used > max_samples:
        for t in sorted(tasks, key=lambda x: alloc[x], reverse=True):
            if used <= max_samples:
                break
            if alloc[t] > 0:
                alloc[t] -= 1
                used -= 1
    remaining = max_samples - used

    sampled = []
    for t in tasks:
        bucket = by_task[t]
        k = min(alloc[t], len(bucket))
        if k <= 0:
            continue
        idx = torch.randperm(len(bucket), generator=rng).tolist()[:k]
        sampled.extend([bucket[i] for i in idx])

    if remaining > 0:
        leftovers = []
        for t in tasks:
            bucket = by_task[t]
            k = min(alloc[t], len(bucket))
            if k < len(bucket):
                leftovers.extend(bucket[k:])
        if leftovers:
            idx = torch.randperm(len(leftovers), generator=rng).tolist()[:remaining]
            sampled.extend([leftovers[i] for i in idx])

    return sampled[:max_samples]


def build_prompt(tokenizer, input_text):
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return input_text + "\n\n### Response:\n"


def generate_open_task_samples(
    model,
    tokenizer,
    records,
    device,
    batch_size=8,
    max_input_length=512,
):
    model.eval()
    model.to(device)

    samples = []

    by_task = defaultdict(list)
    for rec in records:
        task = rec.get("task", "unknown")
        by_task[task].append(rec)

    for task, task_records in by_task.items():
        max_new_tokens = TASK_MAX_TOKENS.get(task, 50)

        for start in range(0, len(task_records), batch_size):
            batch_records = task_records[start:start + batch_size]
            prompts = [build_prompt(tokenizer, r["input"]) for r in batch_records]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for row, rec in enumerate(batch_records):
                # With left padding, attention_mask.sum() is the number of real tokens,
                # not the index where generation starts. Generation starts after the
                # full (padded) input length.
                prompt_len = int(inputs["input_ids"].shape[1])
                new_tokens = output_ids[row][prompt_len:]
                generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                if task in {"qa", "pii_ner", "pii_detection"}:
                    generated = generated.split("\n")[0].strip()

                samples.append({
                    "group": rec.get("group", "unknown"),
                    "task": task,
                    "input": rec["input"],
                    "target": rec["target"],
                    "generated": generated,
                })

    return samples


def infer_epsilon(checkpoint_path):
    if not checkpoint_path:
        return "baseline"

    match = re.search(r"dp_eps([0-9]+(?:\.[0-9]+)?)", checkpoint_path)
    return match.group(1) if match else "unknown"


def run_evaluation(
    model_key,
    checkpoint_path,
    model,
    tokenizer,
    cfg,
    dataset_type,
    device,
    metrics_max_samples=0,
    metrics_batch_size=8,
    num_preview_samples=5,
    save_all_samples=False,
    sample_seed=0,
    data_dir=None,
):
    data_path = get_data_path(dataset_type, data_dir)
    dataset = MultiTaskDataset(data_path, tokenizer, max_length=cfg.max_seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    print(f"\nEvaluating {dataset_type}: {len(dataset):,} examples")

    loss_result = evaluate_loss(model, dataset, dataloader, device)

    metrics = None
    all_samples = []
    metrics_num_samples = 0

    if dataset_type != "attack_eval":
        records = dataset.records
        if metrics_max_samples and metrics_max_samples > 0:
            records = _stratified_sample_by_task(records, metrics_max_samples, seed=sample_seed)

        metrics_num_samples = len(records)

        label_records = [
            r for r in records
            if r.get("task") in TASK_LABELS
        ]

        open_records = [r for r in records if r.get("task") in GEN_TASKS]
        skipped_records = [
            r
            for r in records
            if r.get("task") not in TASK_LABELS and r.get("task") not in GEN_TASKS
        ]

        print(f"  Label-scored records: {len(label_records):,}")
        print(f"  Generation records  : {len(open_records):,}")
        if skipped_records:
            skipped_tasks = sorted({r.get('task', 'unknown') for r in skipped_records})
            print(f"  Skipped tasks       : {skipped_tasks}")

        label_dataset = RecordsSubset(dataset, label_records)
        label_samples = evaluate_classification_by_loss(
            model=model,
            tokenizer=tokenizer,
            dataset=label_dataset,
            device=device,
            max_samples=len(label_records),
        )

        gen_samples = generate_open_task_samples(
            model=model,
            tokenizer=tokenizer,
            records=open_records,
            device=device,
            batch_size=metrics_batch_size,
            max_input_length=cfg.max_seq_length,
        )

        all_samples = label_samples + gen_samples

        samples_by_task = defaultdict(list)
        for sample in all_samples:
            samples_by_task[sample["task"]].append(sample)

        if os.environ.get("DP_LLM_DEBUG_TASKS"):
            print("tasks in samples_by_task:", sorted(samples_by_task.keys()))

        metrics = compute_all_metrics(dict(samples_by_task))
        if "pii_detection" in samples_by_task:
            metrics.setdefault("pii_detection", {}).update(
                relaxed_pii_scores(samples_by_task["pii_detection"])
            )
        print_metrics_table(
            metrics,
            model_key=model_key,
            epsilon=infer_epsilon(checkpoint_path),
        )

    preview = all_samples[:num_preview_samples]

    return {
        "dataset": dataset_type,
        "num_samples": len(dataset),
        "avg_loss": loss_result["avg_loss"],
        "group_losses": loss_result["group_losses"],
        "task_losses": loss_result["task_losses"],
        "metrics": metrics,
        "metrics_num_samples": metrics_num_samples,
        "samples": all_samples if save_all_samples else preview,
        "eval_modes": {
            "classification": "loss_based_label_scoring",
            "nli": "loss_based_label_scoring",
            "open_output_tasks": "greedy_generation",
            "attack_eval": "loss_only",
        },
    }


def main(
    model_key,
    checkpoint_path,
    dataset_type,
    device="cuda",
    metrics_max_samples=0,
    metrics_batch_size=8,
    num_preview_samples=5,
    save_all_samples=False,
    sample_seed=0,
    data_dir=None,
    results_dir=None,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU")
        device = "cpu"

    model, tokenizer, cfg = load_model_and_tokenizer(model_key, checkpoint_path)

    datasets = list(DATASET_FILES.keys()) if dataset_type == "all" else [dataset_type]

    results = {
        "model_key": model_key,
        "checkpoint": checkpoint_path,
        "device": device,
        "evaluations": [],
    }

    for ds in datasets:
        results["evaluations"].append(
            run_evaluation(
                model_key=model_key,
                checkpoint_path=checkpoint_path,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                dataset_type=ds,
                device=device,
                metrics_max_samples=metrics_max_samples,
                metrics_batch_size=metrics_batch_size,
                num_preview_samples=num_preview_samples,
                save_all_samples=save_all_samples,
                sample_seed=sample_seed,
                data_dir=data_dir,
            )
        )

    results_dir = results_dir or repo_path("results_testing")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_tag = "base"
    if checkpoint_path:
        ckpt_tag = checkpoint_path.strip("/").replace("/", "__")

    out_path = os.path.join(
        results_dir,
        f"{model_key}_{ckpt_tag}_{dataset_type}_clean_eval.json",
    )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", required=True, choices=["test", "val", "attack_eval", "all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--metrics_max_samples", type=int, default=0)
    parser.add_argument("--metrics_batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--save_all_samples", action="store_true")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--results-dir", default=None)

    args = parser.parse_args()

    main(
        model_key=args.model,
        checkpoint_path=args.checkpoint,
        dataset_type=args.dataset,
        device=args.device,
        metrics_max_samples=args.metrics_max_samples,
        metrics_batch_size=args.metrics_batch_size,
        num_preview_samples=args.num_samples,
        save_all_samples=args.save_all_samples,
        sample_seed=args.sample_seed,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
