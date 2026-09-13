#!/usr/bin/env python
"""Fast preflight: does the configured HF token actually have *content* access
to the gated base model? `whoami` and `model_info` can succeed while a
fine-grained token still 403s on file content -- this checks the real thing
(a HEAD on config.json) so a GPU job fails in seconds, not after a queue
wait + partial model download.

Exit 0 = good. Exit 3 = token present but forbidden (prints the fix).
Exit 4 = no token / other error.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from finetune.config import MODELS

MODEL_KEY = os.environ.get("DPLLM_MODEL_KEY", "llama3_2_1b_it")


def main() -> int:
    repo = MODELS[MODEL_KEY].model_name
    tok = (os.environ.get("HF_TOKEN")
           or os.environ.get("HUGGING_FACE_HUB_TOKEN")
           or "").strip()
    if not tok:
        print(f"PREFLIGHT FAIL: no HF token in env (HF_TOKEN). Need one for gated {repo}.")
        return 4

    import requests
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                         allow_redirects=True, timeout=30)
    except Exception as e:
        print(f"PREFLIGHT FAIL: cannot reach huggingface.co: {e!r}")
        return 4

    if r.status_code == 200:
        who = "?"
        try:
            who = requests.get("https://huggingface.co/api/whoami-v2",
                               headers={"Authorization": f"Bearer {tok}"},
                               timeout=15).json().get("name", "?")
        except Exception:
            pass
        print(f"PREFLIGHT OK: token ({who}) has content access to {repo}")
        return 0

    print(f"PREFLIGHT FAIL: HTTP {r.status_code} on {url}")
    print(f"  {r.text[:300]}")
    if r.status_code in (401, 403):
        print()
        print("  The token authenticates but lacks GATED-REPO CONTENT access. Fix on")
        print("  huggingface.co (any one):")
        print("   1. Edit this fine-grained token -> enable 'Read access to contents of")
        print("      all public gated repos you can access' -> save.")
        print("   2. Or create a classic 'Read' token and overwrite")
        print("      ~/.config/dp-llm/hf_token")
        print(f"   3. Or add {repo} to the token's per-repo read list.")
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
