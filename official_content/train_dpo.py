from dotenv import load_dotenv
import argparse
import sys
import os
import random
from typing import List, Dict, Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import mlflow

# コマンドライン引数でstageを指定
parser = argparse.ArgumentParser(description="DPO training script")
parser.add_argument("--stage", type=int, default=1, help="DPO stage number (1, 2, 3, ...)")
args = parser.parse_args()

STAGE = args.stage
print(f"[INFO] DPO Training Stage: {STAGE}")

# 指定されたstageのenvファイルを読み込む
stage_env_path = f"/root/workspace/2025_llm_comp_main/official_content/dpo_stage{STAGE}.env"
if not os.path.exists(stage_env_path):
    print(f"[ERROR] DPO stage env file not found: {stage_env_path}")
    sys.exit(1)

load_dotenv(stage_env_path)
print(f"[INFO] Loaded env from: {stage_env_path}")

# -----------------------------
# Config (env-overridable)
# -----------------------------

def _getenv(name: str, default: str):
    return os.environ.get(name, default)

def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

# ベースモデル
BASE_MODEL_ID = _getenv("DPO_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# SFTアダプタのパス（マージして開始する場合に指定）
SFT_ADAPTER_DIR = _getenv("DPO_SFT_ADAPTER_DIR", "")

# DPOデータセット
DATASET_ID = _getenv("DPO_DATASET_ID", "u-10bei/dpo-dataset-qwen-cot")

# 出力ディレクトリ
OUT_DIR = _getenv("DPO_OUT_DIR", "/root/workspace/2025_llm_comp_main/output/dpo_lora_model")
OUT_DIR = f"{OUT_DIR}_stage{STAGE}"

# シーケンス長
MAX_SEQ_LEN = _getenv_int("DPO_MAX_SEQ_LEN", 2048)

# LoRA Config
LORA_R = _getenv_int("DPO_LORA_R", 8)
LORA_ALPHA = _getenv_int("DPO_LORA_ALPHA", 16)
LORA_DROPOUT = _getenv_float("DPO_LORA_DROPOUT", 0)
LORA_TARGET_MODULES = (
    _getenv("DPO_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
)

# DPO Hyperparams
LR = _getenv_float("DPO_LR", 1e-7)
BETA = _getenv_float("DPO_BETA", 0.1)
NUM_TRAIN_EPOCHS = _getenv_int("DPO_EPOCHS", 1)
PER_DEVICE_TRAIN_BATCH_SIZE = _getenv_int("DPO_PER_DEVICE_TRAIN_BS", 2)
GRAD_ACCUM = _getenv_int("DPO_GRAD_ACCUM", 4)
MAX_LENGTH = _getenv_int("DPO_MAX_LENGTH", 1024)
MAX_PROMPT_LENGTH = _getenv_int("DPO_MAX_PROMPT_LENGTH", 512)
WARMUP_RATIO = _getenv_float("DPO_WARMUP_RATIO", 0.1)
WEIGHT_DECAY = _getenv_float("DPO_WEIGHT_DECAY", 0.01)
SEED = _getenv_int("DPO_SEED", 42)
MAX_STEPS = _getenv_int("DPO_MAX_STEPS", -1)

# Logging / Save
LOGGING_STEPS = _getenv_int("DPO_LOGGING_STEPS", 50)
SAVE_STEPS = _getenv_int("DPO_SAVE_STEPS", 100)
SAVE_TOTAL_LIMIT = _getenv_int("DPO_SAVE_TOTAL_LIMIT", 2)


# -----------------------------
# Seed
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# -----------------------------
# DPO Dataset Formatting
# -----------------------------

def format_dpo_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """prompt/chosen/rejected をチャットテンプレート形式に整形する"""

    def formatting_prompts_func(examples):
        new_prompts = []
        new_chosens = []
        new_rejecteds = []

        for prompt, chosen, rejected in zip(
            examples["prompt"], examples["chosen"], examples["rejected"]
        ):
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_chosen = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": chosen}],
                tokenize=False,
            )
            formatted_rejected = tokenizer.apply_chat_template(
                [{"role": "assistant", "content": rejected}],
                tokenize=False,
            )

            new_prompts.append(formatted_prompt)
            new_chosens.append(formatted_chosen)
            new_rejecteds.append(formatted_rejected)

        return {
            "prompt": new_prompts,
            "chosen": new_chosens,
            "rejected": new_rejecteds,
        }

    return dataset.map(formatting_prompts_func, batched=True)


# -----------------------------
# Main
# -----------------------------

def main():
    # MLflow設定
    REPO_ROOT = "/root/workspace/2025_llm_comp_main"
    mlflow_tracking_uri = f"file://{REPO_ROOT}/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = _getenv("MLFLOW_EXPERIMENT_NAME", f"dpo-training-stage{STAGE}")
    mlflow.set_experiment(experiment_name)
    print(f"[INFO] MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"[INFO] MLflow experiment: {experiment_name}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- モデルのロード ---
    print(f"[INFO] Loading base model: {BASE_MODEL_ID}")

    # SFTアダプタのマージ処理
    sft_adapter_path = os.path.expanduser(SFT_ADAPTER_DIR) if SFT_ADAPTER_DIR else ""
    if sft_adapter_path and os.path.exists(os.path.join(sft_adapter_path, "adapter_config.json")):
        print(f"[INFO] Merging SFT adapter from: {sft_adapter_path}")

        # bfloat16でロードしてマージ（DPOもbf16で学習するため統一）
        model_16bit, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_ID,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )

        from peft import PeftModel
        model_16bit = PeftModel.from_pretrained(model_16bit, sft_adapter_path)
        print(f"[INFO] Loaded SFT adapter")

        model_merged = model_16bit.merge_and_unload()
        print(f"[INFO] Merged SFT adapter into base model")

        # マージ済みモデルを一時保存
        merged_model_path = f"{OUT_DIR}_merged_base"
        os.makedirs(merged_model_path, exist_ok=True)
        model_merged.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)
        print(f"[INFO] Saved merged model to {merged_model_path}")

        # メモリ解放
        del model_16bit, model_merged
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # マージ済みモデルを4bitでロード
        print(f"[INFO] Reloading merged model in 4bit...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=merged_model_path,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    else:
        if SFT_ADAPTER_DIR:
            print(f"[WARNING] SFT adapter not found at {sft_adapter_path}, starting from base model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_ID,
            max_seq_length=MAX_SEQ_LEN,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )

    # LoRA適用
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    print("[INFO] LoRA adapters applied")

    # チャットテンプレート設定
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # --- データセットのロード ---
    print(f"[INFO] Loading DPO dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"[INFO] Dataset loaded: {len(dataset)} samples")

    # フォーマッティング
    dataset = format_dpo_dataset(dataset, tokenizer)
    print(f"[INFO] Dataset formatted")
    print(f"[INFO] Sample prompt (truncated): {dataset[0]['prompt'][:200]}...")

    # --- DPO Training ---
    dpo_config = DPOConfig(
        output_dir=OUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        beta=BETA,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        seed=SEED,
        report_to="mlflow",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles this for PEFT
        processing_class=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
    )
    print("[INFO] DPOTrainer initialized")

    print("[INFO] Starting DPO training...")
    trainer_stats = trainer.train()

    # 保存
    print("[INFO] Saving model & tokenizer...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[INFO] Done. Saved to {OUT_DIR}")
    print(f"[INFO] Training stats: {trainer_stats}")


if __name__ == "__main__":
    main()
