#!/bin/bash
set -e

echo "Setting up training environment..."

# Create training virtual environment
if [ ! -d ".venv-train" ]; then
    echo "Creating training virtual environment (.venv-train)..."
    uv venv .venv-train
else
    echo "Training virtual environment already exists..."
fi

# Activate training environment
echo "Activating training environment..."
source .venv-train/bin/activate

# Install training dependencies (excluding eval group)
echo "Installing training dependencies..."
uv sync --active --no-group eval

echo ""
echo "=========================================="
echo "Training environment setup complete!"
echo "=========================================="
echo ""
echo "To use the training environment:"
echo "  source .venv-train/bin/activate"
echo ""
