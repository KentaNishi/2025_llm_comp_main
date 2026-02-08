#!/bin/bash
set -e

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

# Run the evaluation
echo "Running evaluation script..."
python3 /home/nkutm/workspace/2025-llm-advance-competition-main/official_content/evaluation.py "$@"

echo "Evaluation complete!"
