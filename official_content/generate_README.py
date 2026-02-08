# -----------------------------
# README.md（モデルカード）を OUT_LORA_DIR に生成
# -----------------------------
# 学習完了後に実行し、Hugging Face の README.md（モデルカード）を生成
# ベースモデル名・データセット名・学習ハイパーパラメータはコードの変数から自動同期

import os
from dotenv import load_dotenv

# 学習コードと同じ env ファイルを読み込む
load_dotenv("execution.env")

# ------------------------------------------------------------------
# 補助関数の定義
# ------------------------------------------------------------------
def _s(x, default=""):
    try:
        v = str(x)
        return v if v.strip() else default
    except Exception:
        return default

def _fmt_lr(x) -> str:
    """
    Learning Rate の表記を整えるための関数。

    - 数値として解釈できる場合：
      指数表記（例: 1e-6）に整形する
    - 数値として解釈できない場合：
      元の値をそのまま文字列として出力する
      （誤った値を生成しないための安全策）
    """
    try:
        return f"{float(x):.0e}"
    except Exception:
        return _s(x, "")


# ------------------------------------------------------------------
# execution.env から環境変数を読み込み（README と自動同期）
# ------------------------------------------------------------------
base_model_id = _s(os.environ.get('SFT_BASE_MODEL', ''))
dataset_id_raw = _s(os.environ.get('SFT_DATASET_ID', ''))
dataset_id = f"https://huggingface.co/datasets/{dataset_id_raw}" if dataset_id_raw else ""
OUT_LORA_DIR = os.environ.get('SFT_OUT_LORA_DIR', './lora_output')

max_seq_len = int(os.environ.get('SFT_MAX_SEQ_LEN', '512'))
epochs = int(os.environ.get('SFT_EPOCHS', '1'))
lr_str = _fmt_lr(os.environ.get('SFT_LR', '1e-6'))

lora_r = int(os.environ.get('SFT_LORA_R', '64'))
lora_alpha = int(os.environ.get('SFT_LORA_ALPHA', '128'))

# NOTE:
# - YAML front matter の license は
#   「この LoRA アダプタ（リポジトリ）のライセンス表明」を意味する。
# - 必要に応じて環境変数で差し替え可能。
repo_license = os.environ.get("SFT_REPO_LICENSE", "apache-2.0")

# README 内に記載するモデルタイトル
# 変更したい場合は README.md を手書きで調整
title_line = input("＜【課題】ここは自分で記入して下さい＞") #例： qwen3-4b-structured-output-lora

# 出力ディレクトリを作成
os.makedirs(OUT_LORA_DIR, exist_ok=True)

# ------------------------------------------------------------------
# README.md 本文の生成
# （説明テキストに準拠し、変数部分のみを自動置換）
# ------------------------------------------------------------------
readme_md = f"""---
base_model: {base_model_id}
datasets:
- {dataset_id}
language:
- en
license: {repo_license}
library_name: peft
pipeline_tag: text-generation
tags:
- qlora
- lora
- structured-output
---

{title_line}

This repository provides a **LoRA adapter** fine-tuned from
**{base_model_id}** using **QLoRA (4-bit, Unsloth)**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.

## Training Objective

This adapter is trained to improve **structured output accuracy**
(JSON / YAML / XML / TOML / CSV).

Loss is applied only to the final assistant output,
while intermediate reasoning (Chain-of-Thought) is masked.

## Training Configuration

- Base model: {base_model_id}
- Method: QLoRA (4-bit)
- Max sequence length: {max_seq_len}
- Epochs: {epochs}
- Learning rate: {lr_str}
- LoRA: r={lora_r}, alpha={lora_alpha}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "{base_model_id}"
adapter = "your_id/your-repo"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & Terms (IMPORTANT)

Training data: {dataset_id}

Dataset License: MIT License. This dataset is used and distributed under the terms of the MIT License.
Compliance: Users must comply with the MIT license (including copyright notice) and the base model's original terms of use.
"""
# ------------------------------------------------------------------
# README.md の書き込み
# ------------------------------------------------------------------

readme_path = os.path.join(OUT_LORA_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_md)

# ------------------------------------------------------------------
# 動作確認
# ------------------------------------------------------------------

assert os.path.exists(readme_path), "README.md was not written."
assert readme_md.lstrip().startswith("---\n"), (
    "README.md must start with YAML front matter."
)
# 修正: 先頭の --- は改行なしで始まるため count("\n---\n") には含まれない。
# そのため、閉じタグの分として 1回以上あればOKとする。
assert readme_md.count("\n---\n") >= 1, (
    "YAML front matter must be closed properly."
)

print(f"[INFO] README.md written to: {readme_path}")
print("[INFO] Preview (first 30 lines):")
for i, line in enumerate(readme_md.splitlines()[:30], start=1):
    print(f"{i:02d}: {line}")