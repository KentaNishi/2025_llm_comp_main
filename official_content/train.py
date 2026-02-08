from dotenv import load_dotenv
import argparse
import sys
import os
import json
import shutil
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import Trainer, TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel
import mlflow

# コマンドライン引数でstageを指定
parser = argparse.ArgumentParser(description="Multi-stage training script")
parser.add_argument("--stage", type=int, default=1, help="Training stage number (1, 2, 3, ...)")
args = parser.parse_args()

STAGE = args.stage
print(f"[INFO] Training Stage: {STAGE}")

# 指定されたstageのenvファイルを読み込む
stage_env_path = f"/home/nkutm/workspace/2025-llm-advance-competition-main/official_content/stage{STAGE}.env"
if not os.path.exists(stage_env_path):
    print(f"[ERROR] Stage env file not found: {stage_env_path}")
    sys.exit(1)

load_dotenv(stage_env_path)
print(f"[INFO] Loaded env from: {stage_env_path}")

# -----------------------------
# 2.1) Config (env-overridable)
# -----------------------------
# “環境変数で上書きできる設定”を用意しています。
# つまり、コードを編集しなくても、Colabの環境変数を変えるだけで
# ベースモデル名、学習率、エポック数などを変更できる設計です。
#
# この設計のメリット：
# - “標準コード”は同じまま、ハイパーパラメータだけ試せる（再現性が高い）

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

# 学習の“出発点”となるベースモデル（4B）
BASE_MODEL_ID = _getenv("SFT_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")

# 学習に使うSFTデータセット（HF Hub上に置かれている想定）
DATASET_ID    = _getenv("SFT_DATASET_ID", "u-10bei/structured_data_with_cot_dataset_512_v2")

# 複数データセットのMix設定（JSON形式）
# 例: [{"id": "dataset1", "weight": 1.0, "split": "train"}, {"id": "dataset2", "weight": 2.0}]
# weightが大きいほど、そのデータセットからより多くサンプリングされます
DATASET_MIX_JSON = _getenv("SFT_DATASET_MIX", "")

# 学習後に保存されるLoRAアダプタの出力先（ローカル）
OUT_LORA_DIR  = _getenv("SFT_OUT_LORA_DIR", "/content/lora_structeval_t_qwen3_4b") # HFアップロードするアダプタ名と合わせる
# stageごとに別のディレクトリに保存
OUT_LORA_DIR = f"{OUT_LORA_DIR}_stage{STAGE}"

SEED        = _getenv_int("SFT_SEED", 3407)
VAL_RATIO   = _getenv_float("SFT_VAL_RATIO", 0.05)

# 1サンプルあたり最大何トークンまで見るか（長いほど情報を見られるが、GPUメモリと時間が増える）
MAX_SEQ_LEN = _getenv_int("SFT_MAX_SEQ_LEN", 512)

# LoRA Config（＝“どれくらいの表現力を持つ差分を学習するか”）
LORA_R       = _getenv_int("SFT_LORA_R", 64)
LORA_ALPHA   = _getenv_int("SFT_LORA_ALPHA", 128)
LORA_DROPOUT = _getenv_float("SFT_LORA_DROPOUT", 0)
LORA_TARGET_MODULES = (
    _getenv("SFT_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
)

# Train hyperparams（学習の基本設定）
NUM_TRAIN_EPOCHS            = _getenv_int("SFT_EPOCHS", 1)
PER_DEVICE_TRAIN_BATCH_SIZE = _getenv_int("SFT_PER_DEVICE_TRAIN_BS", 2)
PER_DEVICE_EVAL_BATCH_SIZE  = _getenv_int("SFT_PER_DEVICE_EVAL_BS", 2)

# 勾配累積：GPUに一度に載せられるバッチが小さい時に、複数ステップ分を貯めて“大きいバッチ相当”にする
GRAD_ACCUM                  = _getenv_int("SFT_GRAD_ACCUM", 8)

LR                          = _getenv_float("SFT_LR", 1e-6)
WARMUP_RATIO                = _getenv_float("SFT_WARMUP_RATIO", 0.1)

# Debug / quick check
# MAX_STEPSを小さくすると“動作確認だけ”の短時間学習ができます（本番は -1 のまま）
MAX_STEPS        = _getenv_int("SFT_MAX_STEPS", -1)
LOGGING_STEPS    = _getenv_int("SFT_LOGGING_STEPS", 10)
EVAL_STEPS       = _getenv_int("SFT_EVAL_STEPS", 50)
SAVE_STEPS       = _getenv_int("SFT_SAVE_STEPS", 100)
SAVE_TOTAL_LIMIT = _getenv_int("SFT_SAVE_TOTAL_LIMIT", 2)
WEIGHT_DECAY     = _getenv_float("SFT_WEIGHT_DECAY", 0.05)

# Optional: upsampling rules
# 特定のサブカテゴリ（例：難しいタスク）を“多めに学習させる”ための仕組み。
# 標準ではOFFになっています。
UPSAMPLE_ENABLE     = _getenv("SFT_USE_UPSAMPLING", "0") in ("1","true","True")
UPSAMPLE_RULES_JSON = _getenv("SFT_UPSAMPLE_RULES", "")


# -----------------------------
# 2.2) Seed & Utils
# -----------------------------
# 乱数（シャッフルやサンプリング）を固定して、再現性を担保します。
# seedが同じなら、原則として同じ分割・同じ抽出になりやすいです。

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

def ensure_openai_messages(ds: Dataset, msg_col: str = "messages") -> None:
    # データが「messages: [{role, content}, ...]」形式かをチェックします。
    # これは ChatGPT形式（OpenAIのChat Completions形式に似た）で、
    # tokenizer.apply_chat_template で安全に文字列化するために必要です。
    row0 = ds[0]
    ex = row0.get(msg_col, None)
    if not isinstance(ex, list):
        raise ValueError(f"Dataset must have list-style 'messages'. Got {type(ex)}")

def has_any_nonempty_assistant_turn(msgs: List[Dict[str, Any]]) -> bool:
    # “assistantの発話が空じゃない”ものが1回でも含まれるか？
    # SFTでは「正解例（assistantの出力）」がないと学習できないため。
    return any(m.get("role") == "assistant" and str(m.get("content", "")).strip() != "" for m in msgs)

def ends_with_nonempty_assistant(ex: Dict[str, Any]) -> bool:
    # 最後のターンが assistant の回答になっているサンプルだけを使います。
    # こうしておくと「最後のassistantだけ学習する（assistant-only loss）」設計と相性が良いです。
    msgs = ex.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    c = msgs[-1].get("content", "")
    return isinstance(c, str) and c.strip() != ""

def shuffle_split(ds: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    # データをシャッフルして train/val に分割します。
    # val（検証）を持つことで「学習が進むほど性能が上がっているか／過学習していないか」を見られます。
    ds_shuf = ds.shuffle(seed=seed)
    n = len(ds_shuf)
    n_val = max(1, int(round(n * val_ratio)))
    return ds_shuf.select(range(n_val, n)), ds_shuf.select(range(n_val))

def make_text_cache_builder(tokenizer):
    # messages形式 → 実際にモデルに入力する“1本のテキスト”へ変換する関数を作ります。さらに「トークン長（truncationなし）」もキャッシュします。
    #
    # full_text  : ユーザー＋アシスタント（正解）まで含んだ全文
    # prefix_text: “最後のassistantの直前まで”の文（＝ここからassistantを生成させたい）
    #
    # この2つを持つことで、後のcollatorで「assistant部分だけをloss対象にする境界」を計算できます。

    def _build(batch):
        full_out = []
        prefix_out = []
        full_len_out = []
        prefix_len_out = []

        for msgs in batch["messages"]:
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

            full_out.append(full)
            prefix_out.append(prefix)

            # 重要：ここで truncation=False で token 長だけ計算してキャッシュする
            # add_special_tokens=False はあなたの現行設計に合わせる（テンプレ側で必要トークンが入る想定）
            full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]

            full_len_out.append(len(full_ids))
            prefix_len_out.append(len(prefix_ids))

        return {
            "full_text": full_out,
            "prefix_text": prefix_out,
            "full_input_ids_len": full_len_out,
            "prefix_input_ids_len": prefix_len_out,
        }

    return _build



# -----------------------------
# 2.3) Collator (assistant-only loss)
# -----------------------------
# collatorは「生のサンプル群 → 学習に必要なテンソル(input_ids/labels等)」に変換する部品です。
#
# ここがこの学習コードの“設計思想”の核心：
# - 入力（user/system）も含めてモデルには読ませる
# - ただし loss（誤差）を計算するのは assistant の出力部分だけ
#
# これにより：
# - 「プロンプトを丸暗記させる」方向に学習が引っ張られにくい
# - “回答の形式”や“出力の正確さ”に学習の力点を置きます。

# 使用データセットによる仕様の違い
# データセット1：Output: が 100% なので CoT マスクが常に動き、Output本体だけ学習
# データセット2：Output: 系ラベルが存在しないため、CoTマスクは発動せず、“出力本体”を学習

# --- CoT mask settings (env overridable) ---
MASK_COT = _getenv("SFT_MASK_COT", "1") in ("1","true","True")
OUTPUT_MARKERS = [s.strip() for s in _getenv(
    "SFT_OUTPUT_MARKERS",
    "Output:,OUTPUT:,Final:,Answer:,Result:,Response:"
).split(",") if s.strip()]
OUTPUT_LEARN_MODE = _getenv("SFT_OUTPUT_LEARN_MODE", "after_marker")  # after_marker / from_marker

@dataclass
class AssistantOnlyCollatorCached:
    tokenizer: Any
    max_length: int = MAX_SEQ_LEN

    def _find_subsequence(self, seq: List[int], sub: List[int]) -> int:
        if not sub or len(sub) > len(seq):
            return -1
        for i in range(0, len(seq) - len(sub) + 1):
            if seq[i:i+len(sub)] == sub:
                return i
        return -1

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer
        full_texts   = [ex["full_text"] for ex in batch]
        prefix_texts = [ex["prefix_text"] for ex in batch]

        old_trunc = getattr(tok, "truncation_side", "right")
        old_pad   = getattr(tok, "padding_side", "right")
        tok.truncation_side = "left"
        tok.padding_side    = "right"

        try:
            full_enc_tr = tok(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )
            input_ids = full_enc_tr["input_ids"]
            attention_mask = full_enc_tr["attention_mask"]
            labels = torch.full_like(input_ids, fill_value=-100)

            full_ids_nt   = tok(full_texts,   return_tensors=None, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
            prefix_ids_nt = tok(prefix_texts, return_tensors=None, padding=False, truncation=False, add_special_tokens=False)["input_ids"]

            marker_token_seqs = []
            if MASK_COT and OUTPUT_MARKERS:
                for m in OUTPUT_MARKERS:
                    mid = tok(m, add_special_tokens=False, truncation=False)["input_ids"]
                    if not mid:
                        continue
                    mid_nl = tok(m + "\n", add_special_tokens=False, truncation=False)["input_ids"]
                    mid_crlf = tok(m + "\r\n", add_special_tokens=False, truncation=False)["input_ids"]
                    marker_token_seqs.append((mid, mid_nl, mid_crlf))

            for i in range(input_ids.size(0)):
                trunc_left = max(0, len(full_ids_nt[i]) - self.max_length)
                boundary = len(prefix_ids_nt[i]) - trunc_left
                full_len_tr = int(attention_mask[i].sum().item())

                # assistant開始が見えていない => 学習対象外（元コード方針を維持）
                if boundary <= 0 or boundary >= full_len_tr:
                    continue

                span_start = boundary
                span_end   = full_len_tr

                # デフォルト：assistant全体を学習（データセット2はここに落ちる）
                learn_start = span_start

                # CoTマスク：Output marker が見つかったときだけ学習開始点を進める（データセット1で発動）
                if MASK_COT and marker_token_seqs:
                    visible_ids = input_ids[i, :full_len_tr].tolist()
                    assistant_ids = visible_ids[span_start:span_end]

                    best_out = None  # (out_pos, after_pos)
                    for mid, mid_nl, mid_crlf in marker_token_seqs:
                        # 改行付き優先
                        p = self._find_subsequence(assistant_ids, mid_nl)
                        if p != -1:
                            out_pos = span_start + p
                            after_pos = out_pos + len(mid_nl)
                        else:
                            p = self._find_subsequence(assistant_ids, mid_crlf)
                            if p != -1:
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid_crlf)
                            else:
                                p = self._find_subsequence(assistant_ids, mid)
                                if p == -1:
                                    continue
                                out_pos = span_start + p
                                after_pos = out_pos + len(mid)

                        if (best_out is None) or (out_pos < best_out[0]):
                            best_out = (out_pos, after_pos)

                    if best_out is not None:
                        out_pos, after_pos = best_out
                        if OUTPUT_LEARN_MODE == "from_marker":
                            learn_start = out_pos
                        else:
                            learn_start = after_pos
                        learn_start = max(span_start, min(learn_start, span_end))

                if learn_start < span_end:
                    labels[i, learn_start:span_end] = input_ids[i, learn_start:span_end]

            labels[attention_mask == 0] = -100
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        finally:
            tok.truncation_side = old_trunc
            tok.padding_side    = old_pad

@torch.no_grad()
def filter_has_supervision(ds, collator):
    keep = []
    for i in range(len(ds)):
        out = collator([ds[i]])
        if (out["labels"][0] != -100).sum().item() > 0:
            keep.append(i)
    return ds.select(keep)


def count_all_masked(ds, collator, n=200, seed=3407):
    rng = random.Random(seed)
    n = min(n, len(ds))
    idxs = [rng.randrange(0, len(ds)) for _ in range(n)]
    all_masked = 0
    for i in idxs:
        out = collator([ds[i]])
        labels = out["labels"][0]
        if (labels != -100).sum().item() == 0:
            all_masked += 1
    print(f"[CHECK] all-masked samples in {n}: {all_masked} ({all_masked/max(1,n):.1%})")


# -----------------------------
# 2.4) Optional upsampling
# -----------------------------
# upsamplingは「特定の種類のデータを多めに学習させる」テクニックです。
# 例：
# - JSONは得意だがYAMLは苦手 → YAML関連サンプルを2倍にする
# - 特定のsubcategoryが点数に効く → そこを厚くする
# ただし、やりすぎると他が弱くなることもあります（トレードオフ）。
# 学習データセットの品質が悪い等の原因で、却って性能が低下することもあります。
# その場合、学習データセットを観察し、追加の前処理が有効であることも多いです。

def load_and_mix_datasets() -> Dataset:
    """
    複数のデータセットをロードして重み付きで混合します。

    DATASET_MIX_JSONが設定されている場合はそれを使用し、
    設定されていない場合は単一のDATASET_IDを使用します。

    Returns:
        Dataset: 混合されたデータセット
    """
    if not DATASET_MIX_JSON or DATASET_MIX_JSON.strip() == "":
        # 単一データセットモード（後方互換性）
        print(f"[INFO] Loading single dataset from HF Hub: {DATASET_ID}")
        return load_dataset(DATASET_ID, split="train")

    # 複数データセットMixモード
    try:
        mix_config = json.loads(DATASET_MIX_JSON)
        if not isinstance(mix_config, list) or len(mix_config) == 0:
            print("[WARNING] Invalid DATASET_MIX format, falling back to single dataset")
            return load_dataset(DATASET_ID, split="train")
    except Exception as e:
        print(f"[WARNING] Failed to parse DATASET_MIX: {e}, falling back to single dataset")
        return load_dataset(DATASET_ID, split="train")

    print(f"[INFO] Loading and mixing {len(mix_config)} datasets...")

    datasets = []
    weights = []

    for i, cfg in enumerate(mix_config):
        if not isinstance(cfg, dict):
            print(f"[WARNING] Skipping invalid config at index {i}: {cfg}")
            continue

        dataset_id = cfg.get("id", None)
        weight = cfg.get("weight", 1.0)
        split = cfg.get("split", "train")

        if not dataset_id:
            print(f"[WARNING] Skipping config without 'id' at index {i}")
            continue

        try:
            weight = float(weight)
            if weight <= 0:
                print(f"[WARNING] Skipping dataset {dataset_id} with non-positive weight {weight}")
                continue
        except Exception:
            print(f"[WARNING] Invalid weight for {dataset_id}, using 1.0")
            weight = 1.0

        print(f"[INFO] Loading dataset {i+1}/{len(mix_config)}: {dataset_id} (split={split}, weight={weight})")

        try:
            ds = load_dataset(dataset_id, split=split)
            print(f"  -> Loaded {len(ds)} samples")
            datasets.append(ds)
            weights.append(weight)
        except Exception as e:
            print(f"[ERROR] Failed to load {dataset_id}: {e}")
            continue

    if len(datasets) == 0:
        print("[ERROR] No datasets loaded successfully, falling back to single dataset")
        return load_dataset(DATASET_ID, split="train")

    if len(datasets) == 1:
        print("[INFO] Only one dataset loaded, returning as-is")
        return datasets[0]

    # 重み付きサンプリングでデータセットを混合
    print(f"[INFO] Mixing {len(datasets)} datasets with weights: {weights}")

    # 各データセットのサイズを計算
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # 目標サンプル数を決定（最大のデータセットサイズを基準）
    max_size = max(len(ds) for ds in datasets)
    target_samples = [int(max_size * w) for w in normalized_weights]

    print(f"[INFO] Target sample counts: {target_samples} (total: {sum(target_samples)})")

    # 各データセットからサンプリング
    mixed_indices = []
    for i, (ds, target) in enumerate(zip(datasets, target_samples)):
        ds_len = len(ds)
        if target <= ds_len:
            # ダウンサンプリング
            sampled_indices = np.random.choice(ds_len, size=target, replace=False)
        else:
            # アップサンプリング
            sampled_indices = np.random.choice(ds_len, size=target, replace=True)

        # データセットインデックスとサンプルインデックスのペアを保存
        mixed_indices.extend([(i, idx) for idx in sampled_indices])

    # シャッフル
    np.random.shuffle(mixed_indices)

    # 混合データセットを構築
    print(f"[INFO] Building mixed dataset with {len(mixed_indices)} samples...")

    # すべてのサンプルを収集
    all_samples = []
    for ds_idx, sample_idx in mixed_indices:
        all_samples.append(datasets[ds_idx][int(sample_idx)])

    # Datasetオブジェクトとして構築
    from datasets import Dataset

    # サンプルから列名を取得
    if len(all_samples) > 0:
        column_names = list(all_samples[0].keys())
        # 各列のデータを収集
        columns_data = {col: [sample[col] for sample in all_samples] for col in column_names}
        mixed_dataset = Dataset.from_dict(columns_data)
    else:
        mixed_dataset = datasets[0].select([])  # 空のデータセット

    print(f"[INFO] Mixed dataset created with {len(mixed_dataset)} samples")

    return mixed_dataset


def apply_upsampling(train_ds: Dataset) -> Dataset:
    if not UPSAMPLE_ENABLE or not UPSAMPLE_RULES_JSON:
        return train_ds
    try:
        rules = json.loads(UPSAMPLE_RULES_JSON)
        if not isinstance(rules, dict) or not rules:
            return train_ds
    except Exception:
        return train_ds

    packs = train_ds["subcategory"] if "subcategory" in train_ds.column_names else [None]*len(train_ds)
    pack_field = train_ds["pack"] if "pack" in train_ds.column_names else [None]*len(train_ds)

    w = []
    for sub, pk in zip(packs, pack_field):
        weight = 1.0
        ssub = str(sub or "")
        spk  = str(pk or "")
        for pat, mult in rules.items():
            try:
                m = float(mult)
            except Exception:
                m = 1.0
            if pat.startswith("pack:"):
                if spk == pat.split(":",1)[1]:
                    weight *= max(0.0, m)
            else:
                if pat in ssub:
                    weight *= max(0.0, m)
        w.append(weight)

    w = np.asarray(w, dtype=np.float64)
    if (w <= 0).all() or w.sum() == 0:
        return train_ds

    p = w / w.sum()
    n = len(train_ds)
    idx = np.random.choice(np.arange(n), size=n, replace=True, p=p)
    print("[UPSAMPLE] rules:", rules)
    return train_ds.select(idx.tolist())


# -----------------------------
# 2.5) Callback (monitor)
# -----------------------------
# 学習中のデバッグ用コールバックです。
# ここでは「labelsのうち、実際にloss対象になっているトークン割合」を時々表示します。
#
# 意味：
# - valid_ratio が極端に小さい → “学習していない”のと同じ（ラベルがほぼ -100）
# - valid_ratio が適度にある → assistant部分にしっかりlossが乗っている
#
# 初学者向けに言うと：
# - これは“学習がちゃんと効いているかの健康診断”です。

class LabelStatsCallback(TrainerCallback):
    def __init__(self, dataset, collator, name="train", every_n_steps=100):
        self.dataset, self.collator, self.name, self.every_n_steps = dataset, collator, name, every_n_steps

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % self.every_n_steps) == 0:
            batch = [self.dataset[random.randint(0, len(self.dataset)-1)] for _ in range(8)]
            out = self.collator(batch)
            valid = (out["labels"] != -100).sum().item()
            total = (out["attention_mask"] == 1).sum().item()
            print(f"\n[LabelStats:{self.name}] step={state.global_step} valid_ratio={valid/max(1,total):.4f}")


# -----------------------------
# 2.6) Main
# -----------------------------
# 学習を実行します。

def main():
    # MLflow設定：リポジトリ内のmlrunsディレクトリに記録
    REPO_ROOT = "/home/nkutm/workspace/2025-llm-advance-competition-main"
    mlflow_tracking_uri = f"file://{REPO_ROOT}/mlruns"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(_getenv("MLFLOW_EXPERIMENT_NAME", f"llm-training-stage{STAGE}"))
    print(f"[INFO] MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"[INFO] MLflow experiment: {mlflow.get_experiment_by_name(_getenv('MLFLOW_EXPERIMENT_NAME', f'llm-training-stage{STAGE}')).name if mlflow.get_experiment_by_name(_getenv('MLFLOW_EXPERIMENT_NAME', f'llm-training-stage{STAGE}')) else 'will be created'}")

    os.makedirs(OUT_LORA_DIR, exist_ok=True)

    # if you used /content/your_id cache dirs etc, remove to avoid confusion
    if os.path.exists("/content/your_id"):
        shutil.rmtree("/content/your_id")

    # データセットのロード（単一または複数の混合）
    ds_all = load_and_mix_datasets()

    # データ形式チェック（messagesがlistであること）
    ensure_openai_messages(ds_all)

    # 学習できるサンプルだけ残す（assistantが空なら教師信号が無い）
    ds_all = ds_all.filter(lambda ex: has_any_nonempty_assistant_turn(ex["messages"]))
    ds_all = ds_all.filter(ends_with_nonempty_assistant)

    # train/val分割
    train_ds, val_ds = shuffle_split(ds_all, VAL_RATIO, SEED)

    # Optional: upsampling by rule（分割後に適用）
    train_ds = apply_upsampling(train_ds)

    print("[INFO] Loading base model:", BASE_MODEL_ID)

    # ベースモデルをロード
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,  # RTX 5080などの新しいGPUではbfloat16を使用
        load_in_4bit=True,
    )

    # 前のstageのアダプターがある場合はマージしてから新しいLoRAを追加
    # この方法により、前のstageの学習結果を引き継ぎます
    if STAGE > 1:
        prev_adapter_dir = OUT_LORA_DIR.replace(f"stage{STAGE}", f"stage{STAGE-1}")

        if os.path.exists(prev_adapter_dir) and os.path.exists(os.path.join(prev_adapter_dir, "adapter_config.json")):
            print(f"[INFO] Stage {STAGE}: Merging previous adapter from {prev_adapter_dir}")

            # 前のアダプターをロードしてマージ
            # まず16bitでロードし直す必要があるため、一度unloadする
            print(f"[INFO] Reloading model in 16bit for merging...")
            model_16bit, _ = FastLanguageModel.from_pretrained(
                model_name=BASE_MODEL_ID,
                max_seq_length=MAX_SEQ_LEN,
                dtype=torch.float16,
                load_in_4bit=False,
            )

            # 前のアダプターを適用
            from peft import PeftModel
            model_16bit = PeftModel.from_pretrained(model_16bit, prev_adapter_dir)
            print(f"[INFO] Loaded adapter from stage {STAGE-1}")

            # マージしてベースモデルに統合
            print(f"[INFO] Merging adapter into base model...")
            model_merged = model_16bit.merge_and_unload()

            # マージ済みモデルを一時保存
            merged_model_path = f"{OUT_LORA_DIR}_merged_base"
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
                dtype=torch.bfloat16,  # RTX 5080などの新しいGPUではbfloat16を使用
                load_in_4bit=True,
            )
            print(f"[INFO] Successfully loaded merged model for stage {STAGE}")
        else:
            print(f"[WARNING] Previous stage adapter not found at {prev_adapter_dir}, starting from base model")
    else:
        # Stage 1: ベースモデルから開始
        print(f"[INFO] Stage 1: Starting from base model")

    # Cache chat template renders（tokenizerが必要なのでここで初めてbuild_cacheを作る）
    build_cache = make_text_cache_builder(tokenizer)

    train_ds = train_ds.map(build_cache, batched=True, num_proc=1, desc="Caching train")
    val_ds   = val_ds.map(build_cache,   batched=True, num_proc=1, desc="Caching val")

    # Attach LoRA
    # ここで「学習される部分（LoRAアダプタ）」をモデルに追加します。
    # 学習対象は LoRA のパラメータだけになり、ベースモデルの巨大な重みは固定されます。
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )


    # Transformersの引数名がバージョンで揺れることがあります。
    # 今回のバージョンでは eval_strategy を使います。
    args = TrainingArguments(
        output_dir=OUT_LORA_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        max_steps=MAX_STEPS,  # -1 => epoch-based

        bf16=True,            # RTX 5080などの新しいGPUではbfloat16を使用
        fp16=False,

        push_to_hub=False,
        report_to="mlflow",

        group_by_length=False,
        remove_unused_columns=False,
    )

    # assistant-only loss の collator を使う
    collator = AssistantOnlyCollatorCached(tokenizer=tokenizer, max_length=MAX_SEQ_LEN)

    # --- NaN対策：all-masked（教師トークン0）を除去して評価を安定化 ---
    print("[INFO] Checking all-masked samples before filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=SEED)

    print("[INFO] Filtering train/val to remove all-masked samples...")
    train_ds = filter_has_supervision(train_ds, collator)
    val_ds   = filter_has_supervision(val_ds, collator)

    print("[INFO] New sizes:", "train =", len(train_ds), "val =", len(val_ds))
    print("[INFO] Checking all-masked samples after filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=SEED)


    # Trainer（Transformersの標準学習ループ）
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 監視用コールバックを追加（学習が効いているかのヘルスチェック）
    trainer.add_callback(LabelStatsCallback(train_ds, collator, name="train", every_n_steps=LOGGING_STEPS))

    print("[INFO] Starting training...")
    trainer.train()

    # 学習後の保存：LoRAアダプタ＆tokenizer
    print("[INFO] Saving adapter & tokenizer...")
    model.save_pretrained(OUT_LORA_DIR)
    tokenizer.save_pretrained(OUT_LORA_DIR)
    print(f"[INFO] Done. Saved to {OUT_LORA_DIR}")

if __name__ == "__main__":
    main()

