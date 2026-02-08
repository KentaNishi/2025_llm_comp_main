#!/bin/bash
# Wait for v1 training (stages 1 & 2) to complete, then launch v2 parallel training

set -euo pipefail
cd /root/workspace/2025_llm_comp_main

LOG_DIR=logs
STAGE1_LOG="$LOG_DIR/train_stage1.log"
STAGE2_LOG="$LOG_DIR/train_stage2.log"

# MIG UUIDs for devices 0, 1, 2
UUID_0="MIG-c5246f6b-cb53-532e-8b11-84ef00964399"
UUID_1="MIG-5076d4e5-909e-5dd4-b478-f62ceb342102"
UUID_2="MIG-0a09e67b-3d8b-5e73-bf6c-51b3c3e79979"

echo "[$(date)] Waiting for v1 training (stages 1 & 2) to complete..."

# Poll until both stages show DONE
while true; do
    s1_done=$(grep -c "Stage 1 DONE" "$STAGE1_LOG" 2>/dev/null || true)
    s2_done=$(grep -c "Stage 2 DONE" "$STAGE2_LOG" 2>/dev/null || true)
    echo "[$(date)] Stage1 done=$s1_done  Stage2 done=$s2_done"
    if [[ "$s1_done" -ge 1 && "$s2_done" -ge 1 ]]; then
        break
    fi
    sleep 30
done

echo "[$(date)] v1 training complete. Launching v2 parallel training..."

# Launch stage 1 v2 on device 0
(
    export CUDA_VISIBLE_DEVICES="$UUID_0"
    source .venv-train/bin/activate
    echo "[$(date)] Stage 1 v2 START uuid=$UUID_0" | tee "$LOG_DIR/train_stage1_v2.log"
    python official_content/train.py --stage 1 2>&1 | tee -a "$LOG_DIR/train_stage1_v2.log"
    echo "[$(date)] Stage 1 v2 DONE" | tee -a "$LOG_DIR/train_stage1_v2.log"
) &
PID1=$!

# Launch stage 2 v2 on device 1
(
    export CUDA_VISIBLE_DEVICES="$UUID_1"
    source .venv-train/bin/activate
    echo "[$(date)] Stage 2 v2 START uuid=$UUID_1" | tee "$LOG_DIR/train_stage2_v2.log"
    python official_content/train.py --stage 2 2>&1 | tee -a "$LOG_DIR/train_stage2_v2.log"
    echo "[$(date)] Stage 2 v2 DONE" | tee -a "$LOG_DIR/train_stage2_v2.log"
) &
PID2=$!

# Launch stage 3 v2 on device 2
(
    export CUDA_VISIBLE_DEVICES="$UUID_2"
    source .venv-train/bin/activate
    echo "[$(date)] Stage 3 v2 START uuid=$UUID_2" | tee "$LOG_DIR/train_stage3_v2.log"
    python official_content/train.py --stage 3 2>&1 | tee -a "$LOG_DIR/train_stage3_v2.log"
    echo "[$(date)] Stage 3 v2 DONE" | tee -a "$LOG_DIR/train_stage3_v2.log"
) &
PID3=$!

echo "[$(date)] v2 training launched: stage1 PID=$PID1, stage2 PID=$PID2, stage3 PID=$PID3"
echo "[$(date)] Waiting for all v2 stages to complete..."
wait $PID1 $PID2 $PID3
echo "[$(date)] All v2 training complete."
