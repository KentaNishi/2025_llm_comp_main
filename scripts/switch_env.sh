#!/bin/bash

# Environment switcher helper script
# Usage: source scripts/switch_env.sh [train|eval]

if [ "$1" = "train" ]; then
    if [ ! -d ".venv-train" ]; then
        echo "Error: Training environment not found."
        echo "Please run './scripts/setup_train.sh' first."
        return 1
    fi
    echo "Switching to training environment..."
    source .venv-train/bin/activate
    echo "Training environment activated (.venv-train)"

elif [ "$1" = "eval" ]; then
    if [ ! -d ".venv-eval" ]; then
        echo "Error: Evaluation environment not found."
        echo "Please run './scripts/setup_eval.sh' first."
        return 1
    fi
    echo "Switching to evaluation environment..."
    source .venv-eval/bin/activate
    echo "Evaluation environment activated (.venv-eval)"

else
    echo "Usage: source scripts/switch_env.sh [train|eval]"
    echo ""
    echo "Available environments:"
    echo "  train - Training environment (.venv-train)"
    echo "  eval  - Evaluation environment (.venv-eval)"
    echo ""
    echo "Example:"
    echo "  source scripts/switch_env.sh train"
    return 1
fi
