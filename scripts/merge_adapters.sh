#!/bin/bash
# 全ステージのアダプタを順次マージして最終モデルを生成するスクリプト
# 使い方: ./scripts/merge_adapters.sh [--venv .venv-train]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PATH=""

while [[ "$1" == --* ]]; do
    case "$1" in
        --venv=*) VENV_PATH="${1#*=}"; shift ;;
        --venv)   VENV_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$VENV_PATH" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
fi

cd "$BASE_DIR"

source official_content/stage1.env 2>/dev/null || true
BASE_MODEL="${SFT_BASE_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUT_BASE="${SFT_OUT_LORA_DIR:-~/workspace/2025_llm_comp_main/output/lora_structeval_t_qwen3_4b}"
OUT_BASE="${OUT_BASE/#\~/$HOME}"
FINAL_DIR="${OUT_BASE}_final_merged"

echo "=========================================="
echo " Adapter Merge Pipeline"
echo "=========================================="
echo " Base model : $BASE_MODEL"
echo " Output dir : $FINAL_DIR"
echo "=========================================="

python - <<PYEOF
import os, gc, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "$BASE_MODEL"
out_base = "$OUT_BASE"
final_dir = "$FINAL_DIR"

stages = []
for i in range(1, 4):
    adapter_dir = f"{out_base}_stage{i}"
    # checkpoint サブディレクトリも探す
    if not os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        # 最新のチェックポイントを探す
        ckpts = sorted(Path(adapter_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            adapter_dir = str(ckpts[-1])
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        stages.append((i, adapter_dir))
        print(f"[INFO] Stage {i}: {adapter_dir}")
    else:
        print(f"[WARN] Stage {i}: adapter not found at {out_base}_stage{i}, skipping")

if not stages:
    raise RuntimeError("No adapters found. Run training first.")

current_base = base_model_id
for i, (stage_num, adapter_dir) in enumerate(stages):
    print(f"\\n[INFO] === Merging stage {stage_num} adapter ===")
    print(f"[INFO] Loading base: {current_base}")
    model = AutoModelForCausalLM.from_pretrained(
        current_base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(current_base, trust_remote_code=True)
    print(f"[INFO] Applying adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    if i < len(stages) - 1:
        # 中間マージ: 次のステージのベースとして一時保存
        tmp_dir = f"{out_base}_stage{stage_num}_merged_tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        print(f"[INFO] Saved intermediate merge to {tmp_dir}")
        current_base = tmp_dir
    else:
        # 最終マージ: 最終出力先に保存
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[INFO] Saved final model to {final_dir}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

print(f"\\n[OK] All done. Final model at: {final_dir}")
PYEOF
