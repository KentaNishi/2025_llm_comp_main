#!/bin/bash
# 学習完了を監視し、完了後に自動的に評価を実行するスクリプト

TRAIN_LOG="train_stage1.log"
CHECK_INTERVAL=300  # 5分ごとにチェック

echo "================================================"
echo "Training completion monitor started"
echo "Log file: $TRAIN_LOG"
echo "Checking every $CHECK_INTERVAL seconds..."
echo "================================================"

while true; do
    # 訓練プロセスが実行中か確認
    if ! pgrep -f "python train.py --stage 1" > /dev/null; then
        echo ""
        echo "[$(date)] Training process not found - checking if completed..."

        # ログファイルに完了メッセージがあるか確認
        if grep -q "Done. Saved to" "$TRAIN_LOG" 2>/dev/null; then
            echo "[$(date)] ✓ Training completed successfully!"
            echo ""
            echo "================================================"
            echo "Starting evaluation..."
            echo "================================================"

            cd /home/nkutm/workspace/2025-llm-advance-competition-main
            bash scripts/run_eval.sh 2>&1 | tee official_content/evaluation.log

            echo ""
            echo "================================================"
            echo "Evaluation completed!"
            echo "Check results at: official_content/evaluation.log"
            echo "================================================"
            break
        else
            echo "[$(date)] Training may have failed. Check $TRAIN_LOG"
            break
        fi
    fi

    # 現在の進捗を表示
    PROGRESS=$(tail -n 5 "$TRAIN_LOG" 2>/dev/null | grep -oP '\d+/1500' | tail -1)
    if [ -n "$PROGRESS" ]; then
        echo "[$(date)] Training progress: $PROGRESS steps"
    fi

    sleep $CHECK_INTERVAL
done
