# ------------------------------------------------------------
# 1) Config
# ------------------------------------------------------------

MODEL_SOURCE = "adapter_merge"   # "merged" | "base" | "adapter_merge"
# どのモデルを使うかを選びます。今回は、基本的に"adapter_merge"を選んでください。

#   - "base"        : ベースモデル（学習していない素のモデル）
#   - "merged"      : すでにLoRAをマージ済みのモデル（完成品として配布されている想定）
#   - "adapter_merge": ベースモデル + LoRAアダプタをその場で読み込み、ローカルでマージしてから使う

# base model (HF repo id or local path)
# 学習時に使用したベースモデルを入れてください。
BASE_MODEL_ID_OR_PATH   = "Qwen/Qwen3-4B-Instruct-2507"

# merged model (HF repo id or local path)
# アダプタではなくマージモデルをアップロードした場合は、ここにIDをいれてください。
# "merged"を選択した場合に記入
MERGED_MODEL_ID_OR_PATH = "your_id/your-merged-repo"

# adapter merge
# あなたがHuggingFaceにアップロードしたアダプタのIDを入れてください。
# "adapter_merge"を選択した場合に記入
# ローカルの学習済みアダプターを使用
ADAPTER_ID       = "/root/workspace/2025_llm_comp_main/output/lora_structeval_t_qwen3_4b_stage3/checkpoint-51"

# merge済モデルの一時保存
MERGED_LOCAL_DIR = "./merged_model"

# 入力（150問）と出力（提出用）ファイルパスの指定
INPUT_PATH  = "/root/workspace/2025_llm_comp_main/official_content/public_150.json"
OUTPUT_PATH = "/root/workspace/2025_llm_comp_main/outputs/stage3.json"


TEMPERATURE = 0.0
#   0.0 は最も決定的（同じ入力なら同じ出力になりやすい）で、評価用途では一般に安定します。


# ------------------------------------------------------------
# 2) Stable vLLM env (IMPORTANT: must be set BEFORE importing vllm)
# ------------------------------------------------------------

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# vLLM内部でワーカープロセスを作る方式を "spawn" に固定します。
# Colabなど一部環境では "fork" より安定しやすいことがあります。

os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
# vLLMのログレベル（INFO）を設定します。デバッグ時に有用です。

# ------------------------------------------------------------
# 3) Resolve model_path
# ------------------------------------------------------------
# 選んだMODEL_SOURCEに応じて、最終的にvLLMに渡す「モデルの場所(model_path)」を決めます。

def resolve_model_path():
    # どのモデルを使うかに応じて、vLLMへ渡すパス/IDを返す関数

    if MODEL_SOURCE == "base":
        return BASE_MODEL_ID_OR_PATH

    if MODEL_SOURCE == "merged":
        return MERGED_MODEL_ID_OR_PATH

    if MODEL_SOURCE == "adapter_merge":
        # NOTE: torch/CUDA（GPU）を触るため、vLLMを起動する前に済ませます。
        import os, gc
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        print("[INFO] Merging adapter into base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID_OR_PATH,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        # ベースモデルに対応するトークナイザを読み込み（マージ後も同じものを使うのが通常）
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID_OR_PATH, trust_remote_code=True)

        # base_model に LoRAアダプタ(ADAPTER_ID) をマージ
        # merge後はLoRA層を外せるので（unload）、推論時の扱いが単純になります。
        model_to_merge = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        merged_model = model_to_merge.merge_and_unload()

        os.makedirs(MERGED_LOCAL_DIR, exist_ok=True)
        merged_model.save_pretrained(MERGED_LOCAL_DIR)
        tokenizer.save_pretrained(MERGED_LOCAL_DIR)

        del base_model, model_to_merge, merged_model
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] Merged model saved:", MERGED_LOCAL_DIR)
        return MERGED_LOCAL_DIR

    raise ValueError("MODEL_SOURCE must be 'merged'|'base'|'adapter_merge'")

# 最終的に使うモデルのパス/IDを確定
model_path = resolve_model_path()
print("[INFO] Using model:", model_path)

# ------------------------------------------------------------
# 4) Load public_150 and build prompts (no torch usage here)
# ------------------------------------------------------------
# 入力ファイルを読み込み、各問題の「プロンプト（モデルに渡す文字列）」を作ります。

import json
from pathlib import Path
from transformers import AutoTokenizer

pub = json.loads(Path(INPUT_PATH).read_text(encoding="utf-8"))

assert isinstance(pub, list), "public_150.json must be a list"
assert len(pub) == 150, f"public_150 must have 150 items, got {len(pub)}"
assert len({x["task_id"] for x in pub}) == 150, "public_150 has duplicate task_id"

# Safety: ensure output_type exists (office enriched file)

missing_ot = [x.get("task_id") for x in pub if not (x.get("output_type") or "").strip()]

if missing_ot:
    raise RuntimeError(f"FATAL: public_150 missing output_type (not enriched). Examples: {missing_ot[:5]}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# task_ids: 出力に使う task_id の並びを保存
# prompts:   vLLMに渡すプロンプト文字列を保存
task_ids, prompts = [], []

for item in pub:
    task_ids.append(item["task_id"])
    query = item.get("query", "")
    messages = [{"role": "user", "content": query}]
    prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    # ↑ apply_chat_template で「モデルが期待する会話形式の文字列」に整形
    #   tokenize=False : まだトークン化せず、文字列として返す
    #   add_generation_prompt=True : 「ここからアシスタントが答える」境界を追加
    #   これにより、モデルが回答を続けて生成しやすい形になります。

# ------------------------------------------------------------
# 5) Presets + fallback plan
# ------------------------------------------------------------
# vLLM起動時に「文脈長(max_model_len)」や「出力上限(max_tokens)」を大きくしすぎると、
# GPUメモリ不足(OOM)で落ちやすいです。
# そこで、成功しやすい設定をいくつか用意し、失敗したら段階的に軽くして再試行します。
# merged（既に焼き込み済み）と adapter_merge（その場でマージ）では、
# 実メモリ使用量が変わることがあるため、最初に試す設定（gpu_memなど）を変えています。
# 事前に「試行候補リスト」を作り、上から順に試します。

def build_try_configs():

    # Primary presets

    if MODEL_SOURCE == "merged":
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.85},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.80},
        ]
        # ↑ 4096トークンまでの文脈/出力を許しつつ、GPU使用率を0.85→0.80で試す

    elif MODEL_SOURCE == "adapter_merge":
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.60},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.65},
        ]
        # ↑ adapter_merge はメモリが厳しくなりがちなので、gpu_memを低めから試します。

    else:  # base
        base = [
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.80},
            {"max_model_len": 4096, "max_tokens": 4096, "gpu_mem": 0.70},
        ]
        # ↑ baseモデルは比較的軽い想定で、0.80→0.70を試します。

    # Fallback ladder (reduce context / output)
    # 失敗したときの「段階的に軽くする設定」。
    # max_model_len と max_tokens を下げると、必要メモリが減り成功しやすくなります。
    ladder = [
        {"max_model_len": 3072, "max_tokens": 3072},
        {"max_model_len": 2048, "max_tokens": 2048},
        {"max_model_len": 1536, "max_tokens": 1536},
    ]

    # Expand base configs with ladder and a couple gpu_mem tweaks
    # ↑ base設定に対し、ladder段階を「合成」して試行パターンを増やします。
    #   また、gpu_memも少し増やす版を試します（失敗理由が「確保不足」系のときに効く場合がある）。
    out = []
    for cfg in base:
        out.append(cfg)

        for step in ladder:
            out.append({**cfg, **step})

        # try a slightly higher gpu_mem if still failing (some failures are "not enough alloc")
        out.append({**cfg, "gpu_mem": min(0.90, cfg["gpu_mem"] + 0.05)})

    # Deduplicate while preserving order
    # ↑ 似た設定が重複し得るので、順序を保ったまま重複削除します。
    seen = set()
    uniq = []
    for c in out:
        key = (c["max_model_len"], c["max_tokens"], round(c["gpu_mem"], 2))

        if key in seen:
            continue

        seen.add(key)
        uniq.append(c)

    return uniq


TRY_CONFIGS = build_try_configs()
# ↑ 実際に試す設定リストを作成します。

# ------------------------------------------------------------
# 6) vLLM run with retry
# ------------------------------------------------------------
# ↑ ここからが推論本体です。

# vLLM は CUDA_VISIBLE_DEVICES に MIG UUID を受け付けないため整数インデックスに変換
_cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _cv.startswith("MIG-"):
    import subprocess as _sp
    _lines = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    for _i, _line in enumerate(_lines):
        if _cv in _line:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(_i)
            print(f"[INFO] CUDA_VISIBLE_DEVICES: {_cv} -> {_i}")
            break

from vllm import LLM, SamplingParams
def run_with_config(cfg):

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=cfg["max_tokens"],
    )

    llm = LLM(
        model=model_path,
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=cfg["gpu_mem"],
        enforce_eager=True,
        tensor_parallel_size=1,
         disable_log_stats=True,
    )

    outs = llm.generate(prompts, sampling)

    submission = []
    # ↑ 提出形式 [{"task_id": ..., "generation": ...}, ...] を作ります。

    for tid, out in zip(task_ids, outs):
        gen = out.outputs[0].text if out.outputs else ""
        submission.append({"task_id": tid, "generation": gen})
    return submission
    # ↑ 150問ぶんの提出配列を返します。

if __name__ == "__main__":
    print("[INFO] Try configs (in order):")

    for i, c in enumerate(TRY_CONFIGS[:8], 1):
        print(f"  {i:02d}. max_model_len={c['max_model_len']} max_tokens={c['max_tokens']} gpu_mem={c['gpu_mem']}")

    if len(TRY_CONFIGS) > 8:
        print(f"  ... total {len(TRY_CONFIGS)} configs")

    last_err = None
    submission = None
    # ↑ 成功した場合に提出データ（150件）を入れる変数。成功まではNone。

    for idx, cfg in enumerate(TRY_CONFIGS, 1):
        print(f"[INFO] Attempt {idx}/{len(TRY_CONFIGS)}: max_model_len={cfg['max_model_len']} max_tokens={cfg['max_tokens']} gpu_mem={cfg['gpu_mem']}")
        try:
            submission = run_with_config(cfg)
            print("[INFO] ✅ Generation succeeded with this config.")
            # ↑ 成功ログ
            break
        except RuntimeError as e:
            last_err = e
            msg = str(e)
            print("[WARN] Failed:", msg[:200].replace("\n", " "))

    # try next config
    if submission is None:
        raise RuntimeError(f"All configs failed. Last error: {last_err}")


    # Final guards
    # ↑ 最後に「提出物としての整合性チェック」をします。

    if len(submission) != 150:
        # ↑ 150件生成できているかチェック
        raise RuntimeError(f"Submission count mismatch: {len(submission)}")

    if len({x['task_id'] for x in submission}) != 150:
        # ↑ task_id の重複がないかチェック
        raise RuntimeError("Duplicate task_id in submission")

    Path(OUTPUT_PATH).write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    # ↑ submission（Pythonオブジェクト）をJSON文字列にしてファイルへ保存します。

    print("[OK] wrote:", OUTPUT_PATH, "items=150")
