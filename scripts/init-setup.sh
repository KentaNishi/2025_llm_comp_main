#!/bin/bash
set -e

echo "=========================================="
echo "Initial Instance Setup"
echo "=========================================="

# 1. システムパッケージのインストール
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y git gh python3.12-dev

# 2. Git の設定
echo "[2/7] Configuring git..."
git config --global user.email "nkutm2438@outlook.jp"
git config --global user.name "Kenta Nishi"
echo "  user.name:  $(git config --global user.name)"
echo "  user.email: $(git config --global user.email)"

# 3. uv (Python パッケージマネージャ) のインストール
echo "[3/7] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
else
    echo "  uv already installed: $(uv --version)"
fi

# 4. Claude Code CLI のインストール
echo "[4/7] Installing Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    curl -fsSL https://claude.ai/install.sh | bash
else
    echo "  claude already installed"
fi

# 5. リポジトリのクローン
echo "[5/7] Cloning repository..."
mkdir -p ~/workspace
cd ~/workspace
if [ ! -d "2025_llm_comp_main" ]; then
    git clone https://github.com/KentaNishi/2025_llm_comp_main.git
else
    echo "  Repository already exists, pulling latest..."
    cd 2025_llm_comp_main && git pull && cd ..
fi
cd 2025_llm_comp_main

# 6. 訓練環境のセットアップ
echo "[6/7] Setting up training environment..."
./scripts/setup_train.sh

# 7. 評価環境のセットアップ
echo "[7/7] Setting up evaluation environment..."
./scripts/setup_eval.sh

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "動作確認:"
echo "  .venv-train/bin/python -c \"import torch; print('CUDA:', torch.cuda.is_available(), 'Arch:', torch.cuda.get_device_capability())\""
echo ""
echo "訓練の開始:"
echo "  .venv-train/bin/python ./official_content/train.py --stage 1"
echo ""
echo "評価の実行:"
echo "  ./scripts/run_eval.sh"
echo ""
