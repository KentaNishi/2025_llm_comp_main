"""
MIG device 3 で動くマージ＋評価スクリプト (v2)

対象モデル:
  - v2 SFT Stage 3 (checkpoint-100): ベースSFTモデル
  - DPO Stage 1: SFT Stage3上のDPO学習

マージ戦略:
  1. SFT単体 / DPO単体 (ベースライン)
  2. Linear: SFT + DPO の加重平均 (複数比率)
  3. SLERP: 球面線形補間 (複数t)
  4. Task Arithmetic: ベースからの差分合成
  5. TIES-Merging: Top-k delta + 符号一致フィルタ
  6. DARE: ランダムドロップ + リスケール

※ マージ済みモデルはディスクに保存せず、メモリ上で直接評価する
"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"

import gc
import json
import time
import tomllib
import csv
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Optional

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
OUTPUT_DIR = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/outputs/merge_eval_v2"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# アダプタ定義
ADAPTER_CONFIGS = {
    "sft_v2_s3": {
        "base": BASE_MODEL_ID,
        "adapter": OUTPUT_ROOT / "lora_structeval_t_qwen3_4b_v2_stage3" / "checkpoint-100",
    },
    "dpo_s1": {
        "base": BASE_MODEL_ID,
        "sft_adapter": OUTPUT_ROOT / "lora_structeval_t_qwen3_4b_v2_stage3" / "checkpoint-100",
        "adapter": OUTPUT_ROOT / "dpo_base_model",  # 最新チェックポイントを自動検出
    },
}


# ============================================================
# ユーティリティ
# ============================================================
def find_latest_checkpoint(model_dir: Path) -> Path:
    """ディレクトリから最新のチェックポイントを見つける"""
    if (model_dir / "adapter_config.json").exists():
        return model_dir
    ckpts = sorted(model_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    for ckpt in reversed(ckpts):
        if (ckpt / "adapter_config.json").exists():
            return ckpt
    return model_dir


def wait_for_dpo(timeout=3600, interval=30):
    """DPOの最低1チェックポイントを待機"""
    dpo_dir = ADAPTER_CONFIGS["dpo_s1"]["adapter"]
    print(f"[WAIT] DPO チェックポイント待機中: {dpo_dir}")
    start = time.time()
    while time.time() - start < timeout:
        ckpt = find_latest_checkpoint(dpo_dir)
        if (ckpt / "adapter_config.json").exists():
            print(f"  Found: {ckpt}")
            return ckpt
        elapsed = int(time.time() - start)
        print(f"  まだ見つかりません ({elapsed}s 経過) — {interval}s 後に再確認")
        time.sleep(interval)
    raise TimeoutError("DPOチェックポイント待機タイムアウト")


# ============================================================
# アダプタ → state_dict (ディスク保存なし)
# ============================================================
def merge_single_adapter(base_id: str, adapter_path: Path) -> dict:
    """ベースモデル + 単一アダプタ → state_dict (メモリ上、ディスク保存なし)"""
    print(f"  Loading base: {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    print(f"  Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    del model; gc.collect(); torch.cuda.empty_cache()
    return state_dict


def merge_dpo_adapter(base_id: str, sft_adapter: Path, dpo_adapter: Path) -> dict:
    """ベース + SFTアダプタマージ → DPOアダプタ適用 → state_dict (メモリ上、ディスク保存なし)"""
    print(f"  Loading base + SFT adapter: {sft_adapter}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(sft_adapter))
    model = model.merge_and_unload()
    print(f"  Applying DPO adapter: {dpo_adapter}")
    model = PeftModel.from_pretrained(model, str(dpo_adapter))
    model = model.merge_and_unload()
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    del model; gc.collect(); torch.cuda.empty_cache()
    return state_dict


# ============================================================
# マージ戦略
# ============================================================
def linear_merge(models: list, weights: list) -> dict:
    """加重平均マージ"""
    total = sum(weights)
    weights = [w / total for w in weights]
    keys = list(models[0].keys())
    result = {}
    for key in keys:
        result[key] = sum(w * m[key].float() for m, w in zip(models, weights))
        result[key] = result[key].to(models[0][key].dtype)
    return result


def slerp(w1: dict, w2: dict, t: float) -> dict:
    """SLERP: 球面線形補間"""
    result = {}
    for key in w1:
        a = w1[key].float()
        b = w2[key].float()
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
    """Task Arithmetic: base + Σ coeff_i * (model_i - base)"""
    result = {k: v.clone().float() for k, v in base.items()}
    for model_w, coeff in zip(models, coeffs):
        for key in base:
            result[key] += coeff * (model_w[key].float() - base[key].float())
    return {k: v.to(base[k].dtype) for k, v in result.items()}


def ties_merge(base: dict, models: list, coeffs: list, top_k: float = 0.2) -> dict:
    """TIES-Merging: trim, elect sign, disjoint merge"""
    deltas = []
    for model_w, coeff in zip(models, coeffs):
        delta = {}
        for key in base:
            d = coeff * (model_w[key].float() - base[key].float())
            threshold = torch.quantile(d.abs().float(), 1.0 - top_k)
            d[d.abs() < threshold] = 0.0
            delta[key] = d
        deltas.append(delta)

    result = {k: v.clone().float() for k, v in base.items()}
    for key in base:
        stacked = torch.stack([d[key] for d in deltas])
        sign_sum = stacked.sign().sum(dim=0)
        majority_sign = sign_sum.sign()
        merged = torch.zeros_like(stacked[0])
        for d in deltas:
            mask = (d[key].sign() == majority_sign) & (d[key] != 0)
            merged += d[key] * mask.float()
        nonzero_count = sum((d[key] != 0).float() for d in deltas)
        nonzero_count = nonzero_count.clamp(min=1)
        merged /= nonzero_count
        result[key] += merged

    return {k: v.to(base[k].dtype) for k, v in result.items()}


def dare_merge(base: dict, models: list, coeffs: list, drop_rate: float = 0.9) -> dict:
    """DARE: Drop And REscale"""
    torch.manual_seed(42)
    result = {k: v.clone().float() for k, v in base.items()}

    for model_w, coeff in zip(models, coeffs):
        for key in base:
            delta = model_w[key].float() - base[key].float()
            mask = (torch.rand_like(delta) > drop_rate).float()
            rescaled = delta * mask / (1.0 - drop_rate)
            result[key] += coeff * rescaled

    return {k: v.to(base[k].dtype) for k, v in result.items()}


# ============================================================
# フォーマットバリデーション
# ============================================================
def validate_format(text: str, output_type: str) -> tuple:
    import re
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
            if not rows: return False, "empty CSV"
            return True, ""
        else:
            return True, f"unknown type: {output_type}"
    except Exception as e:
        return False, str(e)[:120]


# ============================================================
# 推論 (transformers, メモリ上の weights から直接ロード)
# ============================================================
def run_inference(weights: dict, label: str) -> dict:
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
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True))

    # state_dict からモデルを構築 (ディスク書き込みなし)
    print(f"  [transformers] {label}: モデルロード中...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.load_state_dict(weights, strict=False)
    model = model.cuda()

    texts = []
    batch_size = 4
    total = len(prompts)
    for start in range(0, total, batch_size):
        batch = prompts[start:start + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                     max_length=2048).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        for i, ids in enumerate(out_ids):
            prompt_len = inputs["input_ids"][i].shape[0]
            generated = tok.decode(ids[prompt_len:], skip_special_tokens=True)
            texts.append(generated)
        done = min(start + batch_size, total)
        print(f"  [transformers] {done}/{total} 完了", end="\r")

    print()
    del model; gc.collect(); torch.cuda.empty_cache()

    # バリデーション集計
    results = []
    valid_count = 0
    type_stats = {}
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
    print(f"\n[RESULT] {label}: {score:.1f}% valid ({valid_count}/{len(pub)})")
    for t, s in sorted(type_stats.items()):
        print(f"  {t:6s}: {s['ok']}/{s['ok']+s['fail']}")

    out_file = OUTPUT_DIR / f"{label}.json"
    out_file.write_text(json.dumps({"summary": summary, "details": results},
                                    ensure_ascii=False, indent=2))
    return summary


# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 60)
    print(" モデルマージ＆評価パイプライン v2")
    print(f" {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(" ※ マージモデルはディスクに保存しません")
    print("=" * 60)

    # ========================================
    # Step 1: アダプタ → state_dict (メモリ上)
    # ========================================
    print("\n[STEP 1] アダプタ → state_dict (メモリ上)")

    # SFT v2 Stage 3
    sft_cfg = ADAPTER_CONFIGS["sft_v2_s3"]
    print(f"\n  SFT v2 Stage3:")
    sft_w = merge_single_adapter(sft_cfg["base"], sft_cfg["adapter"])

    # DPO (チェックポイントを待機)
    dpo_cfg = ADAPTER_CONFIGS["dpo_s1"]
    dpo_ckpt = find_latest_checkpoint(dpo_cfg["adapter"])
    if not (dpo_ckpt / "adapter_config.json").exists():
        print("\n  DPO チェックポイント待機中...")
        dpo_ckpt = wait_for_dpo(timeout=3600, interval=30)

    print(f"\n  DPO Stage1 (adapter: {dpo_ckpt}):")
    dpo_w = merge_dpo_adapter(dpo_cfg["base"], dpo_cfg["sft_adapter"], dpo_ckpt)

    # ========================================
    # Step 2: ベースライン評価
    # ========================================
    print("\n[STEP 2] ベースライン評価")
    summaries = []

    print("\n  評価: sft_v2_s3")
    summaries.append(run_inference(sft_w, "sft_v2_s3"))

    print("\n  評価: dpo_s1")
    summaries.append(run_inference(dpo_w, "dpo_s1"))

    # ========================================
    # Step 3: マージ戦略
    # ========================================
    print("\n[STEP 3] マージ戦略で候補モデルを生成")

    print("  ベースモデルをロード中...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    base_w = {k: v for k, v in base_model.state_dict().items()}
    del base_model; gc.collect()

    # --- Strategy A: Linear (SFT重視) ---
    print("\n  [A] Linear merge: SFT 0.7 + DPO 0.3")
    merged = linear_merge([sft_w, dpo_w], [0.7, 0.3])
    summaries.append(run_inference(merged, "linear_sft07_dpo03"))

    # --- Strategy B: Linear (均等) ---
    print("\n  [B] Linear merge: SFT 0.5 + DPO 0.5")
    merged = linear_merge([sft_w, dpo_w], [0.5, 0.5])
    summaries.append(run_inference(merged, "linear_equal"))

    # --- Strategy C: Linear (DPO重視) ---
    print("\n  [C] Linear merge: SFT 0.3 + DPO 0.7")
    merged = linear_merge([sft_w, dpo_w], [0.3, 0.7])
    summaries.append(run_inference(merged, "linear_sft03_dpo07"))

    # --- Strategy D: SLERP t=0.3 ---
    print("\n  [D] SLERP: SFT → DPO, t=0.3")
    merged = slerp(sft_w, dpo_w, t=0.3)
    summaries.append(run_inference(merged, "slerp_t03"))

    # --- Strategy E: SLERP t=0.5 ---
    print("\n  [E] SLERP: SFT → DPO, t=0.5")
    merged = slerp(sft_w, dpo_w, t=0.5)
    summaries.append(run_inference(merged, "slerp_t05"))

    # --- Strategy F: Task Arithmetic ---
    print("\n  [F] Task Arithmetic: SFT 1.0 + DPO 0.3")
    merged = task_arithmetic(base_w, [sft_w, dpo_w], [1.0, 0.3])
    summaries.append(run_inference(merged, "task_arith_sft10_dpo03"))

    # --- Strategy G: TIES-Merging ---
    print("\n  [G] TIES-Merging: top_k=0.2, SFT 1.0 + DPO 0.5")
    merged = ties_merge(base_w, [sft_w, dpo_w], [1.0, 0.5], top_k=0.2)
    summaries.append(run_inference(merged, "ties_sft10_dpo05"))

    # --- Strategy H: DARE ---
    print("\n  [H] DARE: drop=0.9, SFT 1.0 + DPO 0.3")
    merged = dare_merge(base_w, [sft_w, dpo_w], [1.0, 0.3], drop_rate=0.9)
    summaries.append(run_inference(merged, "dare_sft10_dpo03"))

    del base_w, sft_w, dpo_w, merged; gc.collect()

    # ========================================
    # Step 4: 最終サマリー
    # ========================================
    print("\n" + "=" * 60)
    print(" 最終比較")
    print("=" * 60)
    summaries.sort(key=lambda x: -x["score"])
    for s in summaries:
        print(f"  {s['label']:30s}  {s['score']:5.1f}%  ({s['valid']}/{s['total']})")

    best = summaries[0]
    print(f"\n[BEST] {best['label']}  score={best['score']:.1f}%")
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"[OK] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
