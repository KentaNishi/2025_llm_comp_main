"""
MIG device 3 で動くマージ＋評価スクリプト
- 各ステージの最新チェックポイントを待ってアダプタをマージ
- Linear / SLERP / Task Arithmetic の 3 戦略で候補モデルを生成
- 各候補に対して public_150.json へ推論し、形式バリデーションを実施
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
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 設定
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
OUT_BASE = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/output/lora_structeval_t_qwen3_4b"))
MERGE_OUT_DIR = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/output/merge_candidates"))
INPUT_PATH = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/official_content/public_150.json"))
OUTPUT_DIR = Path(os.path.expanduser("~/workspace/2025_llm_comp_main/outputs/merge_eval"))

MERGE_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# チェックポイント検索
# ============================================================
def find_latest_adapter(stage: int) -> Optional[Path]:
    """ステージの最新チェックポイントを返す"""
    stage_dir = Path(str(OUT_BASE) + f"_stage{stage}")
    # ルート直下に adapter_config.json があれば優先
    if (stage_dir / "adapter_config.json").exists():
        return stage_dir
    # checkpoint-N サブディレクトリから最新を探す
    ckpts = sorted(stage_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    for ckpt in reversed(ckpts):
        if (ckpt / "adapter_config.json").exists():
            return ckpt
    return None

def wait_for_checkpoints(min_stages=1, timeout=7200, interval=30):
    """指定数以上のステージにチェックポイントが揃うまで待機"""
    print(f"[WAIT] {min_stages} stage(s) のチェックポイントを待機中...")
    start = time.time()
    while time.time() - start < timeout:
        available = {}
        for s in [1, 2, 3]:
            p = find_latest_adapter(s)
            if p:
                available[s] = p
        if len(available) >= min_stages:
            for s, p in available.items():
                print(f"  Stage {s}: {p}")
            return available
        elapsed = int(time.time() - start)
        print(f"  揃ったステージ: {list(available.keys())} ({elapsed}s 経過) — {interval}s 後に再確認")
        time.sleep(interval)
    raise TimeoutError("チェックポイント待機タイムアウト")

# ============================================================
# モデルマージ
# ============================================================
def load_and_merge_adapter(base_id: str, adapter_path: Path, out_path: Path):
    """ベースモデル + アダプタをマージして out_path に保存"""
    if (out_path / "config.json").exists():
        print(f"  [SKIP] already merged: {out_path}")
        return
    print(f"  Loading base: {base_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    print(f"  Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"  Saved → {out_path}")

def get_state_dict_cpu(model_path: Path) -> dict:
    """モデルの weight を CPU float32 で取得"""
    from safetensors.torch import load_file
    weights = {}
    for f in sorted(model_path.glob("*.safetensors")):
        weights.update(load_file(f, device="cpu"))
    if not weights:
        import torch
        for f in sorted(model_path.glob("pytorch_model*.bin")):
            weights.update(torch.load(f, map_location="cpu"))
    return weights

def save_merged_weights(weights: dict, ref_path: Path, out_path: Path):
    """マージ済み重みを ref_path の構造を使って保存"""
    from safetensors.torch import save_file
    out_path.mkdir(parents=True, exist_ok=True)
    # config / tokenizer をコピー
    import shutil
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json",
                  "chat_template.jinja", "vocab.json", "merges.txt"]:
        src = ref_path / fname
        if src.exists():
            shutil.copy2(src, out_path / fname)
    # 重みを保存 (一括 safetensors)
    save_file({k: v.contiguous() for k, v in weights.items()},
              out_path / "model.safetensors")
    # generation_config があればコピー
    if (ref_path / "generation_config.json").exists():
        shutil.copy2(ref_path / "generation_config.json",
                     out_path / "generation_config.json")
    print(f"  Saved merged weights → {out_path}")

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

def task_arithmetic(base: dict, deltas: list, coeffs: list) -> dict:
    """Task Arithmetic: base + Σ coeff_i * (model_i - base)"""
    result = {k: v.clone().float() for k, v in base.items()}
    for delta_weights, coeff in zip(deltas, coeffs):
        for key in base:
            result[key] += coeff * (delta_weights[key].float() - base[key].float())
    return {k: v.to(base[k].dtype) for k, v in result.items()}

def linear_merge(models: list, weights: list) -> dict:
    """加重平均マージ"""
    assert abs(sum(weights) - 1.0) < 1e-6
    keys = list(models[0].keys())
    result = {}
    for key in keys:
        result[key] = sum(w * m[key].float() for m, w in zip(models, weights))
        result[key] = result[key].to(models[0][key].dtype)
    return result

# ============================================================
# フォーマットバリデーション
# ============================================================
def validate_format(text: str, output_type: str) -> tuple[bool, str]:
    """出力テキストから構造部分を抽出してパース"""
    # フェンスコードブロックを除去
    import re
    text = text.strip()
    # コードフェンス内を取得
    m = re.search(r"```(?:\w+)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    try:
        if output_type == "JSON":
            json.loads(text)
            return True, ""
        elif output_type == "YAML":
            yaml.safe_load(text)
            return True, ""
        elif output_type == "XML":
            ET.fromstring(text)
            return True, ""
        elif output_type == "TOML":
            tomllib.loads(text)
            return True, ""
        elif output_type == "CSV":
            rows = list(csv.reader(StringIO(text)))
            if not rows:
                return False, "empty CSV"
            return True, ""
        else:
            return True, f"unknown type: {output_type}"
    except Exception as e:
        return False, str(e)[:120]

# ============================================================
# vLLM 推論
# ============================================================
def run_inference(model_path: str, label: str) -> dict:
    """vLLM で public_150.json に対して推論し結果を返す"""
    from vllm import LLM, SamplingParams
    pub = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    prompts, task_ids, output_types = [], [], []
    for item in pub:
        task_ids.append(item["task_id"])
        output_types.append(item.get("output_type", ""))
        msgs = [{"role": "user", "content": item.get("query", "")}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                add_generation_prompt=True))

    # vLLM ロード (メモリ節約設定)
    llm = LLM(model=model_path, max_model_len=2048,
              gpu_memory_utilization=0.85, enforce_eager=True,
              disable_log_stats=True)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0,
                                                    max_tokens=2048))
    del llm; gc.collect(); torch.cuda.empty_cache()

    # バリデーション集計
    results = []
    valid_count = 0
    type_stats = {}
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        ok, err = validate_format(text, output_types[i])
        valid_count += ok
        t = output_types[i]
        if t not in type_stats:
            type_stats[t] = {"ok": 0, "fail": 0}
        type_stats[t]["ok" if ok else "fail"] += 1
        results.append({"task_id": task_ids[i], "output": text,
                         "output_type": t, "valid": ok, "error": err})

    score = valid_count / len(pub) * 100
    summary = {"label": label, "model": model_path,
               "score": score, "valid": valid_count,
               "total": len(pub), "by_type": type_stats}
    print(f"\n[RESULT] {label}: {score:.1f}% valid ({valid_count}/{len(pub)})")
    for t, s in sorted(type_stats.items()):
        print(f"  {t:6s}: {s['ok']}/{s['ok']+s['fail']}")

    # 結果保存
    out_file = OUTPUT_DIR / f"{label}.json"
    out_file.write_text(json.dumps({"summary": summary, "details": results},
                                    ensure_ascii=False, indent=2))
    return summary

# ============================================================
# メイン処理
# ============================================================
def main():
    print("=" * 60)
    print(" モデルマージ＆評価パイプライン")
    print("=" * 60)

    # 1. チェックポイント待機 (最低1ステージ揃ったら開始)
    available = wait_for_checkpoints(min_stages=1, timeout=14400, interval=60)

    # 2. 各ステージのアダプタをベースモデルにマージ
    print("\n[STEP 1] アダプタ → フルモデルにマージ")
    stage_models = {}
    for stage, adapter_path in available.items():
        out_path = MERGE_OUT_DIR / f"stage{stage}_full"
        print(f"\n  Stage {stage}: {adapter_path}")
        load_and_merge_adapter(BASE_MODEL_ID, adapter_path, out_path)
        stage_models[stage] = out_path

    # 3. マージ候補の生成
    print("\n[STEP 2] マージ戦略で候補モデルを生成")
    summaries = []

    # まず各ステージ単体を評価
    for stage, model_path in stage_models.items():
        label = f"stage{stage}_only"
        print(f"\n  評価: {label}")
        s = run_inference(str(model_path), label)
        summaries.append(s)

    if len(stage_models) >= 2:
        print("\n  [MERGE] 重みをロード中...")
        base_w = get_state_dict_cpu(stage_models[min(stage_models)])
        stage_weights = {s: get_state_dict_cpu(p) for s, p in stage_models.items()}

        stages_list = sorted(stage_weights.keys())

        # Strategy A: Linear 均等平均
        if len(stages_list) >= 2:
            n = len(stages_list)
            merged_w = linear_merge(
                [stage_weights[s] for s in stages_list],
                [1/n] * n
            )
            out_path = MERGE_OUT_DIR / "linear_equal"
            save_merged_weights(merged_w, stage_models[stages_list[-1]], out_path)
            summaries.append(run_inference(str(out_path), "linear_equal"))

        # Strategy B: SLERP (最初と最後のステージ間、t=0.5)
        if len(stages_list) >= 2:
            s1, s2 = stages_list[0], stages_list[-1]
            merged_w = slerp(stage_weights[s1], stage_weights[s2], t=0.5)
            out_path = MERGE_OUT_DIR / f"slerp_s{s1}_s{s2}_t05"
            save_merged_weights(merged_w, stage_models[s2], out_path)
            summaries.append(run_inference(str(out_path),
                                            f"slerp_s{s1}_s{s2}_t05"))

        # Strategy C: Task Arithmetic (後半ステージを強調)
        if len(stages_list) >= 2:
            base_w2 = get_state_dict_cpu(Path(BASE_MODEL_ID)
                                          if Path(BASE_MODEL_ID).exists()
                                          else stage_models[stages_list[0]])
            # 最終ステージの delta を 1.2x でスケール
            coeffs = [0.6] * len(stages_list)
            coeffs[-1] = 1.2
            merged_w = task_arithmetic(base_w, list(stage_weights.values()), coeffs)
            out_path = MERGE_OUT_DIR / "task_arith_boost_last"
            save_merged_weights(merged_w, stage_models[stages_list[-1]], out_path)
            summaries.append(run_inference(str(out_path), "task_arith_boost_last"))

        del base_w, stage_weights; gc.collect()

    # 4. 最終サマリー
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
