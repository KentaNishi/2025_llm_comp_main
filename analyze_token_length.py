#!/usr/bin/env python3
"""
データセットのトークン長分布を分析するスクリプト
"""

import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
from transformers import AutoTokenizer

# トークナイザーをロード（Llama系のトークナイザーを使用）
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
except:
    print("Warning: Could not load tokenizer, using character-based approximation")
    tokenizer = None

def count_tokens(text):
    """テキストのトークン数をカウント"""
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=True))
    else:
        # トークナイザーがない場合は文字数を4で割って近似
        return len(text) // 4

def analyze_messages_length(messages):
    """messagesフィールドからトークン長を計算"""
    # 全メッセージを結合してトークン数をカウント
    full_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # チャット形式をシミュレート: <role>content</role>
        full_text += f"<|{role}|>\n{content}\n"

    return count_tokens(full_text)

def analyze_jsonl_file(file_path):
    """JSONLファイルを解析"""
    lengths = []
    categories = defaultdict(list)

    print(f"  Analyzing {file_path.name}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                if messages:
                    length = analyze_messages_length(messages)
                    lengths.append(length)

                    # カテゴリ情報を取得（あれば）
                    category = data.get("category") or data.get("subcategory")
                    if category:
                        categories[category].append(length)

                    # メタデータからformat情報を取得（あれば）
                    metadata = data.get("metadata", {})
                    if isinstance(metadata, dict):
                        format_type = metadata.get("format")
                        if format_type:
                            categories[f"format_{format_type}"].append(length)

            except json.JSONDecodeError as e:
                print(f"    Warning: JSON decode error at line {line_num}: {e}")
            except Exception as e:
                print(f"    Warning: Error at line {line_num}: {e}")

    return lengths, categories

def analyze_parquet_file(file_path):
    """Parquetファイルを解析"""
    try:
        import pandas as pd
        import pyarrow.parquet as pq

        print(f"  Analyzing {file_path.name}...")

        df = pd.read_parquet(file_path)
        lengths = []
        categories = defaultdict(list)

        for idx, row in df.iterrows():
            messages = row.get("messages")

            if messages:
                length = analyze_messages_length(messages)
                lengths.append(length)

                # メタデータからformat情報を取得
                metadata = row.get("metadata")
                if metadata and isinstance(metadata, dict):
                    format_type = metadata.get("format")
                    if format_type:
                        categories[f"format_{format_type}"].append(length)

        return lengths, categories

    except ImportError:
        print(f"    Warning: pandas/pyarrow not available, skipping {file_path.name}")
        return [], {}
    except Exception as e:
        print(f"    Warning: Error reading parquet: {e}")
        return [], {}

def print_statistics(name, lengths):
    """統計情報を出力"""
    if not lengths:
        print(f"\n{name}: No data")
        return

    lengths_array = np.array(lengths)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Sample count:    {len(lengths):,}")
    print(f"Mean:            {lengths_array.mean():.1f} tokens")
    print(f"Median:          {np.median(lengths_array):.1f} tokens")
    print(f"Std Dev:         {lengths_array.std():.1f} tokens")
    print(f"Min:             {lengths_array.min():.1f} tokens")
    print(f"Max:             {lengths_array.max():.1f} tokens")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(lengths_array, p)
        print(f"  {p:2d}%:           {val:.1f} tokens")

    # 各閾値での切り捨て率を計算
    print(f"\nTruncation rate at different max_seq_len:")
    for threshold in [256, 512, 768, 1024, 1536, 2048]:
        truncated = (lengths_array > threshold).sum()
        rate = truncated / len(lengths) * 100
        print(f"  {threshold:4d} tokens: {rate:5.2f}% would be truncated")

def main():
    datasets_dir = Path("datasets")

    if not datasets_dir.exists():
        print(f"Error: {datasets_dir} not found")
        return

    all_results = {}

    # 各データセットディレクトリを処理
    for dataset_dir in sorted(datasets_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue

        print(f"\n{'#'*60}")
        print(f"Dataset: {dataset_dir.name}")
        print(f"{'#'*60}")

        dataset_lengths = []
        dataset_categories = defaultdict(list)

        # JSONLファイルを処理
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            lengths, categories = analyze_jsonl_file(jsonl_file)
            dataset_lengths.extend(lengths)
            for cat, vals in categories.items():
                dataset_categories[cat].extend(vals)

        # Parquetファイルを処理
        for parquet_file in dataset_dir.glob("**/*.parquet"):
            lengths, categories = analyze_parquet_file(parquet_file)
            dataset_lengths.extend(lengths)
            for cat, vals in categories.items():
                dataset_categories[cat].extend(vals)

        if dataset_lengths:
            all_results[dataset_dir.name] = dataset_lengths
            print_statistics(dataset_dir.name, dataset_lengths)

            # カテゴリ別の統計（サンプル数が多い場合のみ）
            if dataset_categories:
                print(f"\n{'─'*60}")
                print("Category breakdown:")
                print(f"{'─'*60}")
                for cat in sorted(dataset_categories.keys()):
                    cat_lengths = dataset_categories[cat]
                    if len(cat_lengths) >= 10:  # 10サンプル以上ある場合のみ表示
                        print(f"  {cat:30s}: {len(cat_lengths):5d} samples, "
                              f"mean={np.mean(cat_lengths):6.1f}, "
                              f"median={np.median(cat_lengths):6.1f}, "
                              f"95%={np.percentile(cat_lengths, 95):6.1f}")

    # 全体統計
    if all_results:
        print(f"\n\n{'#'*60}")
        print("OVERALL STATISTICS (All Datasets Combined)")
        print(f"{'#'*60}")

        all_lengths = []
        for lengths in all_results.values():
            all_lengths.extend(lengths)

        print_statistics("All Datasets", all_lengths)

        # データセット別サマリー
        print(f"\n{'─'*60}")
        print("Dataset Summary:")
        print(f"{'─'*60}")
        for name, lengths in sorted(all_results.items()):
            print(f"{name:50s}: {len(lengths):6,} samples, "
                  f"mean={np.mean(lengths):6.1f}, "
                  f"95%={np.percentile(lengths, 95):6.1f}")

    # 推奨事項を出力
    if all_lengths:
        print(f"\n\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")

        p95 = np.percentile(all_lengths, 95)
        p99 = np.percentile(all_lengths, 99)

        print(f"\nBased on the analysis:")
        print(f"  - 95% of samples fit within {p95:.0f} tokens")
        print(f"  - 99% of samples fit within {p99:.0f} tokens")
        print(f"\nRecommended MAX_SEQ_LEN:")

        if p95 <= 512:
            print(f"  ✓ 512 tokens: Good choice (covers {(np.array(all_lengths) <= 512).sum() / len(all_lengths) * 100:.1f}% of data)")
        elif p95 <= 768:
            print(f"  ✓ 768 tokens: Recommended (covers {(np.array(all_lengths) <= 768).sum() / len(all_lengths) * 100:.1f}% of data)")
        elif p95 <= 1024:
            print(f"  ✓ 1024 tokens: Recommended (covers {(np.array(all_lengths) <= 1024).sum() / len(all_lengths) * 100:.1f}% of data)")
        else:
            print(f"  ✓ 1536 or 2048 tokens: Consider for full coverage")

        print(f"\n  Current setting (512) covers: {(np.array(all_lengths) <= 512).sum() / len(all_lengths) * 100:.1f}% of data")

if __name__ == "__main__":
    main()
