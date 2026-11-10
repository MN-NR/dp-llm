#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible Python environment for dp-llm on OSC Cardinal.
#
# Builds .venv/ at the repo root: Python 3.11, torch 2.5.1+cu124 (H100 /
# CUDA-13 driver), and the four pipeline-pinned packages from
# EXPERIMENT_LOG.md (transformers 4.57.6, opacus 1.5.4, peft 0.18.1) plus
# the eval/plotting deps.
#
# Safe to run on a login node -- it only downloads wheels, no GPU needed.
# Run once; slurm/env.sh just activates what this produced.
#
# Usage:
#   bash bin/osc_setup_env.sh            # build (fails if .venv exists)
#   bash bin/osc_setup_env.sh --force    # wipe and rebuild
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venv"
TORCH_VERSION="2.5.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"
PYTHON_VERSION="3.11"

if [[ "${1:-}" == "--force" ]]; then
    rm -rf "$VENV"
elif [[ -d "$VENV" ]]; then
    echo "ERROR: $VENV already exists. Re-run with --force to rebuild." >&2
    exit 1
fi

# uv: fast, reproducible, downloads its own standalone CPython so we don't
# depend on which python module happens to be loaded.
if ! command -v uv >/dev/null 2>&1; then
    if [[ ! -x "$HOME/.local/bin/uv" ]]; then
        echo ">> installing uv (user site)"
        python3 -m pip install --user --quiet uv
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> creating venv ($PYTHON_VERSION) at $VENV"
uv venv --python "$PYTHON_VERSION" "$VENV"

echo ">> installing torch $TORCH_VERSION from $TORCH_INDEX"
uv pip install --python "$VENV/bin/python" "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"

echo ">> installing the rest of requirements.txt"
uv pip install --python "$VENV/bin/python" -r "$REPO_ROOT/requirements.txt"

echo ">> freezing transitive lock to slurm/requirements.lock.txt"
uv pip freeze --python "$VENV/bin/python" > "$REPO_ROOT/slurm/requirements.lock.txt"

echo ">> sanity check"
"$VENV/bin/python" - <<'PY'
import torch, transformers, opacus, peft, datasets
print("  torch       ", torch.__version__)
print("  transformers", transformers.__version__)
print("  opacus      ", opacus.__version__)
print("  peft        ", peft.__version__)
print("  datasets    ", datasets.__version__)
print("  cuda built  ", torch.version.cuda, "(runtime check happens on a GPU node)")
PY

echo
echo "OK. Environment ready at $VENV"
echo "Next: set your HF token (see slurm/README.md), then submit slurm/build_dataset.slurm"
