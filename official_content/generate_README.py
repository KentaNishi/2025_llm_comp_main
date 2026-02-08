# -----------------------------
# README.md（モデルカード）を OUT_LORA_DIR に生成
# -----------------------------
# 学習完了後に実行し、Hugging Face の README.md（モデルカード）を生成
# ベースモデル名・データセット名・学習ハイパーパラメータはコードの変数から自動同期
#
# 使用方法:
#   python generate_README.py <adapter_dir>
#   例: python generate_README.py ../lora_output/lora_structeval_t_qwen3_4b
#       python generate_README.py /content/lora_output

import os
import re
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# スクリプトと同じディレクトリの env ファイルを読み込む（実行ディレクトリに依存しない）
_HERE = Path(__file__).parent
load_dotenv(_HERE / "stage1.env")

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


def resolve_adapter_dir(base_dir: Path) -> Path:
    """adapter_config.json があるディレクトリを返す。
    直下になければ checkpoint-\\d+ サブディレクトリの中で最新のものを探す。"""
    if (base_dir / "adapter_config.json").exists():
        return base_dir
    checkpoints = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and re.match(r"^checkpoint-\d+$", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[INFO] adapter_config.json not found in root, using: {latest}")
        return latest
    return base_dir


def get_dataset_license(dataset_id: str) -> str:
    """データセットIDの著者名からライセンスを判定する。"""
    author = dataset_id.split("/")[0] if "/" in dataset_id else ""
    if author == "u-10bei":
        return "MIT License"
    elif author == "daichira":
        return "CC-BY-4.0"
    else:
        return "unknown"


def collect_datasets() -> list[str]:
    """SFT_DATASET_MIX または SFT_DATASET_ID からデータセットIDリストを取得する。"""
    mix_raw = os.environ.get("SFT_DATASET_MIX", "").strip()
    if mix_raw:
        try:
            mix = json.loads(mix_raw)
            ids = [entry["id"] for entry in mix if "id" in entry]
            if ids:
                return ids
        except Exception:
            pass
    single = _s(os.environ.get("SFT_DATASET_ID", ""))
    if single:
        return [single]
    return []


# ------------------------------------------------------------------
# コマンドライン引数からアダプターディレクトリを取得
# ------------------------------------------------------------------
if len(sys.argv) < 2:
    # 引数なしの場合は環境変数から取得
    OUT_LORA_DIR = os.environ.get("SFT_OUT_LORA_DIR", "./lora_output")
else:
    OUT_LORA_DIR = sys.argv[1]

OUT_LORA_DIR = Path(OUT_LORA_DIR).expanduser().resolve()

# checkpoint サブディレクトリを解決（ファイル存在確認のみ。README は親に書く）
adapter_dir = resolve_adapter_dir(OUT_LORA_DIR)

# ディレクトリ名からモデル名部分を抽出
# 例: /path/to/lora_structeval_t_qwen3_4b -> lora_structeval_t_qwen3_4b
model_name = OUT_LORA_DIR.name

# ------------------------------------------------------------------
# 環境変数を読み込み（README と自動同期）
# ------------------------------------------------------------------
base_model_id = _s(os.environ.get("SFT_BASE_MODEL", ""))
datasets = collect_datasets()

# 必須変数の確認
if not base_model_id:
    raise RuntimeError(
        "SFT_BASE_MODEL が設定されていません。stage1.env を確認してください。"
    )
if not datasets:
    raise RuntimeError(
        "SFT_DATASET_ID または SFT_DATASET_MIX が設定されていません。stage1.env を確認してください。"
    )

max_seq_len = int(os.environ.get("SFT_MAX_SEQ_LEN", "512"))
epochs = int(os.environ.get("SFT_EPOCHS", "1"))
lr_str = _fmt_lr(os.environ.get("SFT_LR", "1e-6"))

lora_r = int(os.environ.get("SFT_LORA_R", "64"))
lora_alpha = int(os.environ.get("SFT_LORA_ALPHA", "128"))

# NOTE:
# - YAML front matter の license は
#   「この LoRA アダプタ（リポジトリ）のライセンス表明」を意味する。
# - 必要に応じて環境変数で差し替え可能。
repo_license = os.environ.get("SFT_REPO_LICENSE", "apache-2.0")

# HF repo ID: DLNorb/<モデル名>
hf_repo_id = f"DLNorb/{model_name}"

# README 内に記載するモデルタイトル（ディレクトリ名から自動生成）
title_line = f"# {model_name}"

print(f"[INFO] Adapter directory : {OUT_LORA_DIR}")
print(f"[INFO] Adapter files from: {adapter_dir}")
print(f"[INFO] Model name        : {model_name}")
print(f"[INFO] HF Repo ID        : {hf_repo_id}")
print(f"[INFO] Datasets          : {datasets}")

# 出力ディレクトリを作成
OUT_LORA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# YAML front matter 用データセットリスト（IDのみ。URLは本文に記載）
# ------------------------------------------------------------------
dataset_yaml_lines = "\n".join(
    f"- {ds}" for ds in datasets
) if datasets else ""

# ------------------------------------------------------------------
# Sources & Terms セクション（データセットごとにライセンスを記載）
# ------------------------------------------------------------------
if datasets:
    license_entries = []
    for ds in datasets:
        lic = get_dataset_license(ds)
        url = f"https://huggingface.co/datasets/{ds}"
        license_entries.append(f"- {url}: {lic}")
    license_section = (
        "## Sources & Terms (IMPORTANT)\n\n"
        "Training data:\n"
        + "\n".join(license_entries)
        + "\n\nCompliance: Users must comply with each dataset's license "
        "(including copyright notice) and the base model's original terms of use."
    )
else:
    license_section = (
        "## Sources & Terms (IMPORTANT)\n\n"
        "Training data: (unknown)\n\n"
        "Compliance: Users must comply with each dataset's license "
        "and the base model's original terms of use."
    )

# ------------------------------------------------------------------
# README.md 本文の生成
# （説明テキストに準拠し、変数部分のみを自動置換）
# ------------------------------------------------------------------
readme_md = f"""---
base_model: {base_model_id}
datasets:
{dataset_yaml_lines}
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
adapter = "{hf_repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

{license_section}
"""

# ------------------------------------------------------------------
# README.md の書き込み
# ------------------------------------------------------------------
readme_path = OUT_LORA_DIR / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_md)

# ------------------------------------------------------------------
# 動作確認
# ------------------------------------------------------------------
assert readme_path.exists(), "README.md was not written."
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
