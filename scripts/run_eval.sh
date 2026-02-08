#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if evaluation environment exists
if [ ! -d ".venv-eval" ]; then
    echo "Error: Evaluation environment not found."
    echo "Please run './scripts/setup_eval.sh' first."
    exit 1
fi

# Check if StructEval is installed
if [ ! -d "StructEval" ]; then
    echo "Error: StructEval not found."
    echo "Please run './scripts/setup_eval.sh' first."
    exit 1
fi

# Activate evaluation environment
echo "Activating evaluation environment (.venv-eval)..."
source .venv-eval/bin/activate

# MIG 空きインスタンス自動選択 (MIG 有効時のみ)
if nvidia-smi -L 2>/dev/null | grep -q "MIG"; then
    echo "Running evaluation via mig_run.sh (auto MIG instance selection)..."
    exec "$SCRIPT_DIR/mig_run.sh" --venv .venv-eval ${MIG_OPTS} \
        python ~/workspace/2025_llm_comp_main/official_content/evaluation.py "$@"
else
    echo "Running evaluation script..."
    python ~/workspace/2025_llm_comp_main/official_content/evaluation.py "$@"
fi

echo "Evaluation complete!"
