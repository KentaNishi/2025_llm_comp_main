#!/bin/bash
# DPO学習をMIGインスタンス上で起動するスクリプト
#
# 使い方:
#   ./scripts/launch_dpo.sh [--stage N] [--wait]
#
# 例:
#   ./scripts/launch_dpo.sh                    # Stage 1をデフォルトで起動
#   ./scripts/launch_dpo.sh --stage 1          # Stage 1を起動
#   ./scripts/launch_dpo.sh --stage 1 --wait   # 空きMIGが出るまで待機

set -euo pipefail
cd /root/workspace/2025_llm_comp_main

STAGE=1
WAIT_FLAG=""

# オプション解析
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --wait)
            WAIT_FLAG="--wait"
            shift
            ;;
        *)
            echo "不明なオプション: $1"
            echo "使い方: $0 [--stage N] [--wait]"
            exit 1
            ;;
    esac
done

LOG_DIR=logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/dpo_stage${STAGE}_${TIMESTAMP}.log"

echo "============================================"
echo " DPO Training - Stage ${STAGE}"
echo " $(date)"
echo " Log: $LOG"
echo "============================================"

./scripts/mig_run.sh $WAIT_FLAG --venv .venv-train \
    python official_content/train_dpo.py --stage "$STAGE" 2>&1 | tee "$LOG"

echo "[$(date)] DPO Stage ${STAGE} DONE" | tee -a "$LOG"
