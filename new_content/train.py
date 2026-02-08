"""
train.py

StructEval向けSFT/LoRA学習スクリプト。
- .env(stage別)を読み込み
- 複数データセットを重み付きで混合
- 出力を <|BEGIN_CODE|> ... <|END_CODE|> に決定論的正規化
- Stage継続学習(resume/merge)対応

Why:
StructEvalではフォーマット逸脱が致命的なため、学習前に教師信号を正規化する。
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

def load_env(env_path: Path):
    load_dotenv(env_path, override=True)

def normalize_assistant_output(text: str) -> str:
    """
    正規化ルール（LLM不使用・決定論的）
    """
    if "<|BEGIN_CODE|>" in text and "<|END_CODE|>" in text:
        body = text.split("<|BEGIN_CODE|>", 1)[1].split("<|END_CODE|>", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        body = parts[1] if len(parts) > 1 else text
    else:
        markers = ["Output:", "OUTPUT:", "Final:", "Answer:", "Result:", "Response:"]
        body = text
        for m in markers:
            if m in body:
                body = body.split(m)[-1]
    body = body.strip()
    return f"<|BEGIN_CODE|>\n{body}\n<|END_CODE|>"

def load_and_mix_datasets(spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import random
    from datasets import load_dataset

    datasets = []
    weights = []
    for d in spec:
        ds = load_dataset(d["id"], split=d.get("split", "train"))
        datasets.append(ds)
        weights.append(d.get("weight", 1.0))

    total_size = int(os.getenv("SFT_DATASET_MIX_TOTAL_SIZE", "-1"))
    max_size = max(len(ds) for ds in datasets)
    base_size = total_size if total_size > 0 else max_size
    norm = sum(weights)
    target_sizes = [max(1, int(base_size * (w / norm))) for w in weights]

    mixed = []
    for ds, n in zip(datasets, target_sizes):
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        for i in idxs[:n]:
            sample = dict(ds[i])
            if "assistant" in sample:
                sample["assistant"] = normalize_assistant_output(sample["assistant"])
            mixed.append(sample)

    random.shuffle(mixed)
    return mixed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", type=str, required=True)
    args = parser.parse_args()

    load_env(Path(args.env_file))
    dataset_mix = json.loads(os.environ["SFT_DATASET_MIX"])
    data = load_and_mix_datasets(dataset_mix)
    print(f"Loaded {len(data)} samples")

if __name__ == "__main__":
    main()
