"""
4ステージ (SFT s1/s2/s3 + DPO) の組み合わせマージ探索
--part 1: Device 2 (ベースライン s1/s2 + Linear系)
--part 2: Device 3 (ベースライン s3/dpo + SLERP/TA/TIES)
--part all: 全部実行 (デフォルト)

マージ対象:
  - SFT Stage 1 (v2, checkpoint-100)
  - SFT Stage 2 (v2, checkpoint-100)
  - SFT Stage 3 (v2, checkpoint-100)
  - DPO Stage 1 (checkpoint-505, SFT s3 ベース)

※ マージ済みモデルはディスクに保存せず、メモリ上で直接評価する
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import argparse
import gc
import json
import time
import tomllib
import csv
import re
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path

import torch
import numpy as np
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 設定
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUTPUT_ROOT = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/output"))
INPUT_PATH = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/official_content/public_150.json"))
RESULT_DIR = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/outputs/merge_explore"))

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# アダプタパス (全てベースモデルに対するアダプタ)
ADAPTERS = {
    "sft_s1": OUTPUT_ROOT / "lora_structeval_t_qwen3_4b_v2_stage1" / "checkpoint-100",
    "sft_s2": OUTPUT_ROOT / "lora_structeval_t_qwen3_4b_v2_stage2" / "checkpoint-100",
    "sft_s3": OUTPUT_ROOT / "lora_structeval_t_qwen3_4b_v2_stage3" / "checkpoint-100",
}
# DPOはSFT s3マージ済みベースに対するアダプタ
DPO_ADAPTER = OUTPUT_ROOT / "dpo_base_model" / "checkpoint-505"
DPO_SFT_BASE = ADAPTERS["sft_s3"]


# ============================================================
# アダプタ → state_dict (ディスク保存なし)
# ============================================================
def merge_adapter_to_state_dict(base_id: str, adapter_path: Path) -> dict:
    """ベース + 単一アダプタ → state_dict (メモリ上、ディスク保存なし)"""
    print(f"  Merging: {adapter_path.name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    del model; gc.collect(); torch.cuda.empty_cache()
    return state_dict


def merge_dpo_to_state_dict(base_id: str, sft_adapter: Path, dpo_adapter: Path) -> dict:
    """ベース + SFTアダプタ + DPOアダプタ → state_dict (メモリ上、ディスク保存なし)"""
    print(f"  Merging: SFT({sft_adapter.name}) + DPO({dpo_adapter.name})")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, str(dpo_adapter))
    model = model.merge_and_unload()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    del model; gc.collect(); torch.cuda.empty_cache()
    return state_dict


# ============================================================
# マージ関数群
# ============================================================
def linear_merge(models: list, weights: list) -> dict:
    total = sum(weights)
    weights = [w / total for w in weights]
    result = {}
    for key in models[0]:
        result[key] = sum(w * m[key].float() for m, w in zip(models, weights))
        result[key] = result[key].to(models[0][key].dtype)
    return result


def slerp(w1: dict, w2: dict, t: float) -> dict:
    result = {}
    for key in w1:
        a, b = w1[key].float(), w2[key].float()
        dot = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
        dot = dot.clamp(-1, 1)
        omega = torch.acos(dot)
        if omega.abs() < 1e-6:
            result[key] = ((1 - t) * a + t * b).to(w1[key].dtype)
        else:
            result[key] = (
                (torch.sin((1 - t) * omega) / torch.sin(omega)) * a +
                (torch.sin(t * omega) / torch.sin(omega)) * b
            ).to(w1[key].dtype)
    return result


def task_arithmetic(base: dict, models: list, coeffs: list) -> dict:
    # Use intersection of keys (lm_head.weight may be tied/missing in merged models)
    common_keys = set(base.keys())
    for m in models:
        common_keys &= set(m.keys())
    result = {k: base[k].clone().float() for k in common_keys}
    for model_w, coeff in zip(models, coeffs):
        for key in common_keys:
            result[key] += coeff * (model_w[key].float() - base[key].float())
    return {k: v.to(base[k].dtype) for k, v in result.items()}


def ties_merge(base: dict, models: list, coeffs: list, top_k: float = 0.2) -> dict:
    # Use intersection of keys (lm_head.weight may be tied/missing in merged models)
    common_keys = set(base.keys())
    for m in models:
        common_keys &= set(m.keys())
    deltas = []
    for model_w, coeff in zip(models, coeffs):
        delta = {}
        for key in common_keys:
            d = coeff * (model_w[key].float() - base[key].float())
            threshold = torch.quantile(d.abs().float(), 1.0 - top_k)
            d[d.abs() < threshold] = 0.0
            delta[key] = d
        deltas.append(delta)

    result = {k: base[k].clone().float() for k in common_keys}
    for key in common_keys:
        stacked = torch.stack([d[key] for d in deltas])
        majority_sign = stacked.sign().sum(dim=0).sign()
        merged = torch.zeros_like(stacked[0])
        for d in deltas:
            mask = (d[key].sign() == majority_sign) & (d[key] != 0)
            merged += d[key] * mask.float()
        nonzero_count = sum((d[key] != 0).float() for d in deltas).clamp(min=1)
        merged /= nonzero_count
        result[key] += merged
    return {k: v.to(base[k].dtype) for k, v in result.items()}


# ============================================================
# 推論 (transformers, メモリ上の weights から直接ロード)
# ============================================================
def validate_format(text: str, output_type: str) -> tuple:
    text = text.strip()
    m = re.search(r"```(?:\w+)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        if output_type == "JSON":
            json.loads(text); return True, ""
        elif output_type == "YAML":
            yaml.safe_load(text); return True, ""
        elif output_type == "XML":
            ET.fromstring(text); return True, ""
        elif output_type == "TOML":
            tomllib.loads(text); return True, ""
        elif output_type == "CSV":
            rows = list(csv.reader(StringIO(text)))
            return (True, "") if rows else (False, "empty CSV")
        else:
            return True, f"unknown: {output_type}"
    except Exception as e:
        return False, str(e)[:120]


def run_eval(weights: dict, label: str) -> dict:
    """state_dict からモデルをロードして評価 (ディスク保存なし)"""
    pub = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    prompts, task_ids, output_types = [], [], []
    for item in pub:
        task_ids.append(item["task_id"])
        output_types.append(item.get("output_type", ""))
        msgs = [{"role": "user", "content": item.get("query", "")}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # state_dict からモデルを構築 (ディスク書き込みなし)
    print(f"  [eval] {label}: モデルロード中...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.load_state_dict(weights, strict=False)
    model = model.cuda()

    texts = []
    bs = 4
    for start in range(0, len(prompts), bs):
        batch = prompts[start:start + bs]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=2048).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False,
                                     temperature=None, top_p=None)
        for i, ids in enumerate(out_ids):
            pl = inputs["input_ids"][i].shape[0]
            texts.append(tok.decode(ids[pl:], skip_special_tokens=True))
        print(f"    {min(start+bs, len(prompts))}/{len(prompts)}", end="\r")
    print()

    del model; gc.collect(); torch.cuda.empty_cache()

    # 集計
    results, valid_count, type_stats = [], 0, {}
    for i, text in enumerate(texts):
        ok, err = validate_format(text, output_types[i])
        valid_count += ok
        t = output_types[i]
        if t not in type_stats:
            type_stats[t] = {"ok": 0, "fail": 0}
        type_stats[t]["ok" if ok else "fail"] += 1
        results.append({"task_id": task_ids[i], "output": text,
                         "output_type": t, "valid": ok, "error": err})

    score = valid_count / len(pub) * 100
    summary = {"label": label, "score": score,
               "valid": valid_count, "total": len(pub),
               "by_type": type_stats}
    print(f"  [RESULT] {label}: {score:.1f}% ({valid_count}/{len(pub)})")
    for t, s in sorted(type_stats.items()):
        print(f"    {t:6s}: {s['ok']}/{s['ok']+s['fail']}")

    (RESULT_DIR / f"{label}.json").write_text(
        json.dumps({"summary": summary, "details": results}, ensure_ascii=False, indent=2))
    return summary


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["1", "2", "all"], default="all",
                        help="1=Linear系(Device2), 2=SLERP/TA/TIES(Device3), all=全部")
    args = parser.parse_args()
    part = args.part

    print("=" * 60)
    print(f" マージ探索: SFT s1/s2/s3 + DPO (part={part})")
    print(f" {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(" ※ マージモデルはディスクに保存しません")
    print("=" * 60)

    # ========================================
    # Phase 1: 全アダプタ → state_dict (メモリ上)
    # ========================================
    print("\n[Phase 1] アダプタ → state_dict 変換 (メモリ上)")
    weights = {}
    for name, adapter_path in ADAPTERS.items():
        weights[name] = merge_adapter_to_state_dict(BASE_MODEL_ID, adapter_path)
    weights["dpo"] = merge_dpo_to_state_dict(BASE_MODEL_ID, DPO_SFT_BASE, DPO_ADAPTER)
    print(f"  変換完了: {list(weights.keys())}")

    need_base = part in ("2", "all")
    base_w = None
    if need_base:
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        base_w = {k: v for k, v in base_model.state_dict().items()}
        del base_model; gc.collect()

    summaries = []

    def do_merge(label, merged_w):
        summaries.append(run_eval(merged_w, label))

    # ========================================
    # Part 1: ベースライン(s1,s2) + Linear系
    # ========================================
    if part in ("1", "all"):
        print("\n[Phase 3-1] ベースライン評価 (s1, s2)")
        for name in ["sft_s1", "sft_s2"]:
            summaries.append(run_eval(weights[name], name))

        print("\n[Phase 4-1] Linear マージ戦略")

        # --- 2モデル Linear ---
        print("\n  --- 2モデル Linear ---")

        do_merge("lin_s3_07_dpo_03",
                 linear_merge([weights["sft_s3"], weights["dpo"]], [0.7, 0.3]))

        do_merge("lin_s3_05_dpo_05",
                 linear_merge([weights["sft_s3"], weights["dpo"]], [0.5, 0.5]))

        do_merge("lin_s1_05_s3_05",
                 linear_merge([weights["sft_s1"], weights["sft_s3"]], [0.5, 0.5]))

        do_merge("lin_s2_05_s3_05",
                 linear_merge([weights["sft_s2"], weights["sft_s3"]], [0.5, 0.5]))

        # --- 3モデル Linear (全SFT) ---
        print("\n  --- 3モデル Linear ---")

        do_merge("lin_s1s2s3_equal",
                 linear_merge([weights["sft_s1"], weights["sft_s2"], weights["sft_s3"]],
                              [1, 1, 1]))

        do_merge("lin_s1s2s3_s3heavy",
                 linear_merge([weights["sft_s1"], weights["sft_s2"], weights["sft_s3"]],
                              [0.2, 0.2, 0.6]))

        # --- 4モデル Linear (全SFT + DPO) ---
        print("\n  --- 4モデル Linear ---")

        do_merge("lin_all_equal",
                 linear_merge([weights["sft_s1"], weights["sft_s2"], weights["sft_s3"], weights["dpo"]],
                              [1, 1, 1, 1]))

        do_merge("lin_all_s3dpo_heavy",
                 linear_merge([weights["sft_s1"], weights["sft_s2"], weights["sft_s3"], weights["dpo"]],
                              [0.1, 0.1, 0.4, 0.4]))

        do_merge("lin_all_s3_heavy",
                 linear_merge([weights["sft_s1"], weights["sft_s2"], weights["sft_s3"], weights["dpo"]],
                              [0.1, 0.1, 0.6, 0.2]))

    # ========================================
    # Part 2: ベースライン(s3,dpo) + SLERP/TA/TIES
    # ========================================
    if part in ("2", "all"):
        if base_w is None:
            print("  Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
            base_w = {k: v for k, v in base_model.state_dict().items()}
            del base_model; gc.collect()

        print("\n[Phase 3-2] ベースライン評価 (s3, dpo)")
        for name in ["sft_s3", "dpo"]:
            summaries.append(run_eval(weights[name], name))

        print("\n[Phase 4-2] SLERP / Task Arithmetic / TIES マージ戦略")

        # --- SLERP ---
        print("\n  --- SLERP ---")

        do_merge("slerp_s3_dpo_t03",
                 slerp(weights["sft_s3"], weights["dpo"], t=0.3))

        do_merge("slerp_s3_dpo_t05",
                 slerp(weights["sft_s3"], weights["dpo"], t=0.5))

        do_merge("slerp_s1_s3_t05",
                 slerp(weights["sft_s1"], weights["sft_s3"], t=0.5))

        # --- Task Arithmetic ---
        print("\n  --- Task Arithmetic ---")

        do_merge("ta_all_equal",
                 task_arithmetic(base_w,
                                 [weights["sft_s1"], weights["sft_s2"], weights["sft_s3"], weights["dpo"]],
                                 [0.5, 0.5, 1.0, 0.3]))

        do_merge("ta_s3_dpo",
                 task_arithmetic(base_w,
                                 [weights["sft_s3"], weights["dpo"]],
                                 [1.0, 0.3]))

        do_merge("ta_s3_boost",
                 task_arithmetic(base_w,
                                 [weights["sft_s1"], weights["sft_s2"], weights["sft_s3"]],
                                 [0.3, 0.3, 1.2]))

        # --- TIES-Merging ---
        print("\n  --- TIES-Merging ---")

        do_merge("ties_all_k02",
                 ties_merge(base_w,
                            [weights["sft_s1"], weights["sft_s2"], weights["sft_s3"], weights["dpo"]],
                            [0.5, 0.5, 1.0, 0.3], top_k=0.2))

        do_merge("ties_s3_dpo_k03",
                 ties_merge(base_w,
                            [weights["sft_s3"], weights["dpo"]],
                            [1.0, 0.5], top_k=0.3))

    del weights; gc.collect()
    if base_w is not None:
        del base_w; gc.collect()

    # ========================================
    # Phase 5: 最終比較
    # ========================================
    print("\n" + "=" * 60)
    print(f" 最終比較 (part={part})")
    print("=" * 60)
    summaries.sort(key=lambda x: -x["score"])
    for i, s in enumerate(summaries):
        marker = " ★" if i == 0 else ""
        print(f"  {s['label']:30s}  {s['score']:5.1f}%  ({s['valid']}/{s['total']}){marker}")

    if summaries:
        best = summaries[0]
        print(f"\n[BEST] {best['label']}  score={best['score']:.1f}%")

    suffix = f"_part{part}" if part != "all" else ""
    summary_path = RESULT_DIR / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[OK] Summary → {summary_path}")


if __name__ == "__main__":
    main()
