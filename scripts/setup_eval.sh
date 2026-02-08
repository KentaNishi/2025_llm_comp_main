#!/bin/bash
set -e

echo "Setting up evaluation environment..."

# Create evaluation virtual environment
if [ ! -d ".venv-eval" ]; then
    echo "Creating evaluation virtual environment (.venv-eval)..."
    uv venv .venv-eval
else
    echo "Evaluation virtual environment already exists..."
fi

# Activate evaluation environment
echo "Activating evaluation environment..."
source .venv-eval/bin/activate

# Clone StructEval repository
if [ ! -d "StructEval" ]; then
    echo "Cloning StructEval repository..."
    git clone -b fix-module-not-found-issue-2 https://github.com/Osakana7777777/StructEval.git
else
    echo "StructEval directory already exists, skipping clone..."
fi

# Install evaluation dependencies (excluding other groups)
echo "Installing evaluation dependencies..."
uv sync --active --only-group eval

# Install flash-attn from pre-built wheel (PyTorch 2.9, Python 3.12, CUDA 12.x, CXX11 ABI TRUE)
echo "Installing flash-attn from pre-built wheel..."
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install "$FLASH_ATTN_WHEEL"

# Install StructEval
echo "Installing StructEval..."
cd StructEval
uv pip install -e .
cd ..

# Create outputs directory
echo "Creating outputs directory..."
mkdir -p outputs

echo ""
echo "=========================================="
echo "Evaluation environment setup complete!"
echo "=========================================="
echo ""
echo "To use the evaluation environment:"
echo "  source .venv-eval/bin/activate"
echo ""
echo "Or run evaluation directly with:"
echo "  ./scripts/run_eval.sh"
echo ""
