#!/usr/bin/env python3
"""
XMLエラーを詳しく調査するスクリプト
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

# 出力マーカー
OUTPUT_MARKERS = [
    "Output:",
    "OUTPUT:",
    "Final:",
    "Answer:",
    "Result:",
    "Response:",
]


def extract_output_from_cot(content):
    """CoTから出力部分を抽出"""
    for marker in OUTPUT_MARKERS:
        if marker in content:
            parts = content.split(marker, 1)
            if len(parts) == 2:
                return parts[1].strip(), True
    return content.strip(), False


def get_assistant_content(messages):
    """assistantの応答を取得"""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def inspect_xml_samples(file_path, max_samples=5):
    """XMLエラーサンプルを詳しく調査"""
    print(f"\n{'='*80}")
    print(f"Inspecting: {file_path.name}")
    print(f"{'='*80}")

    error_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)

                # XMLフォーマットのみチェック
                category = data.get("category", "")
                subcategory = data.get("subcategory", "")

                is_xml = False
                if "XML" in category:
                    is_xml = True
                elif subcategory and "_to_xml" in subcategory:
                    is_xml = True

                if not is_xml:
                    continue

                messages = data.get("messages", [])
                assistant_content = get_assistant_content(messages)

                if assistant_content is None:
                    continue

                output, has_cot = extract_output_from_cot(assistant_content)

                # XMLパース試行
                try:
                    ET.fromstring(output)
                    # 成功した場合はスキップ
                except ET.ParseError as e:
                    error_count += 1

                    if error_count <= max_samples:
                        print(f"\n{'─'*80}")
                        print(f"Error #{error_count} at line {line_num}")
                        print(f"Sample ID: {data.get('id', 'unknown')}")
                        print(f"Category: {category}")
                        print(f"Subcategory: {subcategory}")
                        print(f"Has CoT: {has_cot}")
                        print(f"Error: {e}")
                        print(f"\nUser prompt (first 200 chars):")
                        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                        print(user_msg[:200] + "...")
                        print(f"\nAssistant output (first 500 chars):")
                        print(output[:500])
                        print(f"\nOutput length: {len(output)} chars")

                        # 最初の数文字をチェック
                        if output:
                            first_chars = output[:50]
                            print(f"\nFirst 50 chars: {repr(first_chars)}")

                            # 一般的な問題をチェック
                            if not output.strip().startswith('<'):
                                print("⚠ Output doesn't start with '<' - likely not XML!")
                            elif output.strip().startswith('<?xml'):
                                print("✓ Has XML declaration")
                            else:
                                print("⚠ No XML declaration (may be okay)")

                    if error_count >= max_samples:
                        break

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    print(f"\n{'─'*80}")
    print(f"Total XML errors found: {error_count}")
    print(f"{'─'*80}")

    return error_count


def main():
    datasets_dir = Path("datasets")

    # daichira系のデータセットのみ調査（XMLエラーがあるもの）
    xml_error_datasets = [
        "daichira__structured-3k-mix-sft/synthetic_3k_mix.jsonl",
        "daichira__structured-5k-mix-sft/synthetic_5k_mix.jsonl",
        "daichira__structured-hard-sft-4k/synthetic_hard_structured_v1.jsonl",
    ]

    total_errors = 0

    for dataset_path in xml_error_datasets:
        file_path = datasets_dir / dataset_path
        if file_path.exists():
            errors = inspect_xml_samples(file_path, max_samples=3)
            total_errors += errors

    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total XML errors across all datasets: {total_errors}")


if __name__ == "__main__":
    main()
