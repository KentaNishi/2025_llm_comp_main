#!/bin/bash
# 全3ステージを異なるMIGインスタンスで並列実行するスクリプト
#
# MIG配分:
#   Device 0 (MIG-c524...) → Stage 1  (daichira/structured-5k-mix-sft, LoRA r=16)
#   Device 1 (MIG-5076...) → Stage 2  (mixed: hard-4k + 5k-mix, LoRA r=32)
#   Device 2 (MIG-0a09...) → Stage 3  (v5 dataset, LoRA r=32, seq_len=1024)
#   Device 3 (MIG-cb26...) → 空き (評価等に使用可能)
#
# 注意: 各ステージはベースモデルから独立に学習します。
#       前ステージのアダプタが存在すればマージして開始します。

set -euo pipefail
cd /root/workspace/2025_llm_comp_main

# MIG UUIDs
UUID_0="MIG-c5246f6b-cb53-532e-8b11-84ef00964399"
UUID_1="MIG-5076d4e5-909e-5dd4-b478-f62ceb342102"
UUID_2="MIG-0a09e67b-3d8b-5e73-bf6c-51b3c3e79979"

LOG_DIR=logs
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " Parallel Stage Training Launch"
echo " $(date)"
echo "============================================"
echo ""
echo "MIG割り当て:"
echo "  Stage 1 → Device 0 ($UUID_0)"
echo "  Stage 2 → Device 1 ($UUID_1)"
echo "  Stage 3 → Device 2 ($UUID_2)"
echo ""

# Stage 1
(
    export CUDA_VISIBLE_DEVICES="$UUID_0"
    source .venv-train/bin/activate
    LOG="$LOG_DIR/train_stage1_${TIMESTAMP}.log"
    echo "[$(date)] Stage 1 START (uuid=$UUID_0)" | tee "$LOG"
    python official_content/train.py --stage 1 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Stage 1 DONE (success)" | tee -a "$LOG"
    else
        echo "[$(date)] Stage 1 FAILED (exit=$EXIT_CODE)" | tee -a "$LOG"
    fi
) &
PID1=$!

# Stage 2
(
    export CUDA_VISIBLE_DEVICES="$UUID_1"
    source .venv-train/bin/activate
    LOG="$LOG_DIR/train_stage2_${TIMESTAMP}.log"
    echo "[$(date)] Stage 2 START (uuid=$UUID_1)" | tee "$LOG"
    python official_content/train.py --stage 2 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Stage 2 DONE (success)" | tee -a "$LOG"
    else
        echo "[$(date)] Stage 2 FAILED (exit=$EXIT_CODE)" | tee -a "$LOG"
    fi
) &
PID2=$!

# Stage 3
(
    export CUDA_VISIBLE_DEVICES="$UUID_2"
    source .venv-train/bin/activate
    LOG="$LOG_DIR/train_stage3_${TIMESTAMP}.log"
    echo "[$(date)] Stage 3 START (uuid=$UUID_2)" | tee "$LOG"
    python official_content/train.py --stage 3 2>&1 | tee -a "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Stage 3 DONE (success)" | tee -a "$LOG"
    else
        echo "[$(date)] Stage 3 FAILED (exit=$EXIT_CODE)" | tee -a "$LOG"
    fi
) &
PID3=$!

echo "起動完了:"
echo "  Stage 1 PID=$PID1 → $LOG_DIR/train_stage1_${TIMESTAMP}.log"
echo "  Stage 2 PID=$PID2 → $LOG_DIR/train_stage2_${TIMESTAMP}.log"
echo "  Stage 3 PID=$PID3 → $LOG_DIR/train_stage3_${TIMESTAMP}.log"
echo ""
echo "進捗確認: tail -f $LOG_DIR/train_stage*_${TIMESTAMP}.log"
echo "全完了待ち中..."
echo ""

# 全ステージの完了を待機
FAIL=0
wait $PID1 || FAIL=$((FAIL + 1))
echo "[$(date)] Stage 1 finished (PID=$PID1)"

wait $PID2 || FAIL=$((FAIL + 1))
echo "[$(date)] Stage 2 finished (PID=$PID2)"

wait $PID3 || FAIL=$((FAIL + 1))
echo "[$(date)] Stage 3 finished (PID=$PID3)"

echo ""
echo "============================================"
if [ "$FAIL" -eq 0 ]; then
    echo " 全ステージ完了 (成功)"
else
    echo " 完了 (${FAIL}件失敗)"
fi
echo " $(date)"
echo "============================================"

exit $FAIL
