import json, math, os, torch
from datetime import datetime

LOG_FILE = "training_log.jsonl"

def log_step(step, epoch, loss, grad_norm, lr, privacy_engine, delta=1e-5):
    """Call this every training step."""
    try:
        eps = privacy_engine.get_epsilon(delta)
    except:
        eps = None  # if DP not active yet
    
    record = {
        "type": "step",
        "step": step,
        "epoch": round(epoch, 4),
        "loss": round(loss, 4),
        "grad_norm": round(grad_norm, 5),
        "lr": lr,
        "epsilon_spent": eps,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_epoch(epoch, model, tokenizer, member_canaries, 
               reference_canaries, privacy_engine, groupA_val,
               delta=1e-5, device="cuda"):
    """Call this at end of every epoch. This is your paper's data."""
    model.eval()
    
    # --- epsilon (the most critical number) ---
    try:
        eps = privacy_engine.get_epsilon(delta)
    except:
        eps = -1

    # --- canary losses (member vs reference) ---
    member_losses, reference_losses = [], []
    
    with torch.no_grad():
        for canary in member_canaries:
            ids = torch.tensor(canary["token_ids"]).unsqueeze(0).to(device)
            loss = model(ids, labels=ids).loss.item()
            member_losses.append({"id": canary["id"], "loss": loss,
                                   "category": canary["category"],
                                   "reps": canary["repetitions"]})
        
        for canary in reference_canaries:
            ids = torch.tensor(canary["token_ids"]).unsqueeze(0).to(device)
            loss = model(ids, labels=ids).loss.item()
            reference_losses.append({"id": canary["id"], "loss": loss})

    # --- per-token losses on canaries (needed for WBC attack later) ---
    token_level = []
    with torch.no_grad():
        for canary in member_canaries[:50]:  # first 50 is enough for WBC
            ids = torch.tensor(canary["token_ids"]).unsqueeze(0).to(device)
            logits = model(ids).logits[:, :-1, :]
            targets = ids[:, 1:]
            per_tok = torch.nn.CrossEntropyLoss(reduction='none')(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            ).cpu().tolist()
            token_level.append({"id": canary["id"], "token_losses": per_tok})

    # --- Group A utility (quick perplexity proxy) ---
    groupA_losses = []
    with torch.no_grad():
        for batch in groupA_val:  # keep groupA_val small, ~200 examples
            ids = batch["input_ids"].to(device)
            groupA_losses.append(model(ids, labels=ids).loss.item())
    groupA_ppl = math.exp(sum(groupA_losses) / len(groupA_losses))

    record = {
        "type": "epoch",
        "epoch": epoch,
        "epsilon_spent": eps,
        "delta": delta,
        "member_canary_losses": member_losses,
        "reference_canary_losses": reference_losses,
        "token_level_losses": token_level,
        "member_loss_mean": sum(x["loss"] for x in member_losses) / len(member_losses),
        "reference_loss_mean": sum(x["loss"] for x in reference_losses) / len(reference_losses),
        "groupA_perplexity": round(groupA_ppl, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    print(f"[Epoch {epoch}] ε={eps:.3f} | "
          f"member_loss={record['member_loss_mean']:.4f} | "
          f"ref_loss={record['reference_loss_mean']:.4f} | "
          f"groupA_ppl={groupA_ppl:.2f}")
    
    return record


def save_checkpoint(model, epoch, epsilon, model_name, config):
    """Call this when epsilon crosses 3, 5, or 8."""
    path = f"checkpoints/{model_name}_eps{epsilon:.1f}_epoch{epoch}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "epsilon": epsilon,
        "model_name": model_name,
        "lora_state_dict": model.state_dict(),
        "config": config,
    }, path)
    print(f"Saved checkpoint: {path}")