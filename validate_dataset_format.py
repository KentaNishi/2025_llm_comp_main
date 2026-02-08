#!/usr/bin/env python3
"""
データセットの構造化データフォーマットを検証するスクリプト
各フォーマット（JSON、XML、YAML、TOML、CSV）が正しくパース可能かをチェック
"""

import json
import glob
import csv
import io
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
import yaml

# TOMLパーサーのインポート
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Python < 3.11
    except ImportError:
        print("Warning: tomllib/tomli not available, TOML validation will be skipped")
        tomllib = None

# 出力マーカー（execution.envから）
OUTPUT_MARKERS = [
    "Output:",
    "OUTPUT:",
    "Final:",
    "Answer:",
    "Result:",
    "Response:",
]


def extract_output_from_cot(content):
    """
    CoT（思考連鎖）を含むコンテンツから実際の出力部分を抽出

    Returns:
        tuple: (extracted_output, has_marker)
    """
    # マーカーを探す
    for marker in OUTPUT_MARKERS:
        if marker in content:
            # マーカー以降の部分を抽出
            parts = content.split(marker, 1)
            if len(parts) == 2:
                output = parts[1].strip()
                return output, True

    # マーカーが見つからない場合は全体を返す
    return content.strip(), False


def validate_json(content):
    """JSON形式を検証"""
    try:
        json.loads(content)
        return True, None
    except json.JSONDecodeError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_xml(content):
    """XML形式を検証"""
    try:
        ET.fromstring(content)
        return True, None
    except ET.ParseError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_yaml(content):
    """YAML形式を検証"""
    try:
        yaml.safe_load(content)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_toml(content):
    """TOML形式を検証"""
    if tomllib is None:
        return None, "TOML parser not available"

    try:
        tomllib.loads(content)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_csv(content):
    """CSV形式を検証"""
    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)

        # 最低限1行はあるべき
        if len(rows) == 0:
            return False, "Empty CSV"

        # 各行のカラム数が一致しているか確認
        if len(rows) > 1:
            col_count = len(rows[0])
            for i, row in enumerate(rows[1:], 1):
                if len(row) != col_count:
                    return False, f"Inconsistent column count at row {i+1}: expected {col_count}, got {len(row)}"

        return True, None
    except Exception as e:
        return False, str(e)


def get_format_from_data(data):
    """
    データからフォーマットを推定

    Returns:
        str or None: format type (json, xml, yaml, toml, csv) or None
    """
    # メタデータから取得を試みる（最優先）
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        format_type = metadata.get("format")
        if format_type:
            return format_type.lower()

    # サブカテゴリから推定（2番目に優先、"xxx_to_yyy"のyyy部分がターゲット）
    subcategory = data.get("subcategory", "")
    if subcategory:
        # "xxx_to_yyy" 形式からターゲットフォーマットを抽出
        if "_to_" in subcategory:
            target = subcategory.split("_to_")[-1].lower()
            if target in ["json", "xml", "yaml", "toml", "csv"]:
                return target

    # カテゴリから推定（最後の手段、ただしこれは信頼性が低い）
    # 注意: C_XML は必ずしも出力がXMLとは限らない（xml_to_yaml等）
    category = data.get("category", "")
    if category:
        if "JSON" in category:
            return "json"
        elif "XML" in category:
            return "xml"
        elif "YAML" in category:
            return "yaml"
        elif "TOML" in category:
            return "toml"
        elif "CSV" in category:
            return "csv"

    return None


def get_assistant_content(messages):
    """messagesからassistantの応答を取得"""
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def validate_sample(data, file_path):
    """
    1サンプルを検証

    Returns:
        dict: 検証結果
    """
    messages = data.get("messages", [])
    if not messages:
        return {
            "valid": False,
            "error": "No messages field",
            "format": None,
        }

    assistant_content = get_assistant_content(messages)
    if assistant_content is None:
        return {
            "valid": False,
            "error": "No assistant message",
            "format": None,
        }

    # フォーマットを特定
    format_type = get_format_from_data(data)
    if format_type is None:
        return {
            "valid": None,
            "error": "Cannot determine format",
            "format": None,
        }

    # CoTマーカーから出力を抽出
    output, has_cot = extract_output_from_cot(assistant_content)

    # フォーマットに応じて検証
    validators = {
        "json": validate_json,
        "xml": validate_xml,
        "yaml": validate_yaml,
        "toml": validate_toml,
        "csv": validate_csv,
    }

    validator = validators.get(format_type)
    if validator is None:
        return {
            "valid": None,
            "error": f"Unknown format: {format_type}",
            "format": format_type,
        }

    valid, error = validator(output)

    return {
        "valid": valid,
        "error": error,
        "format": format_type,
        "has_cot": has_cot,
        "output_length": len(output),
        "sample_id": data.get("id", "unknown"),
    }


def validate_jsonl_file(file_path):
    """JSONLファイルを検証"""
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "skipped": 0,
        "by_format": defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0}),
        "errors": [],
    }

    print(f"  Validating {file_path.name}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                results["total"] += 1

                validation = validate_sample(data, file_path)

                format_type = validation["format"]
                if format_type:
                    results["by_format"][format_type]["total"] += 1

                if validation["valid"] is True:
                    results["valid"] += 1
                    if format_type:
                        results["by_format"][format_type]["valid"] += 1
                elif validation["valid"] is False:
                    results["invalid"] += 1
                    if format_type:
                        results["by_format"][format_type]["invalid"] += 1

                    # エラーを記録（最初の10個まで）
                    if len(results["errors"]) < 10:
                        results["errors"].append({
                            "line": line_num,
                            "format": format_type,
                            "error": validation["error"],
                            "sample_id": validation.get("sample_id"),
                        })
                else:
                    results["skipped"] += 1

            except json.JSONDecodeError as e:
                print(f"    Warning: JSON decode error at line {line_num}: {e}")
                results["total"] += 1
                results["skipped"] += 1
            except Exception as e:
                print(f"    Warning: Error at line {line_num}: {e}")
                results["total"] += 1
                results["skipped"] += 1

    return results


def validate_parquet_file(file_path):
    """Parquetファイルを検証"""
    try:
        import pandas as pd

        print(f"  Validating {file_path.name}...")

        df = pd.read_parquet(file_path)

        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "skipped": 0,
            "by_format": defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0}),
            "errors": [],
        }

        for idx, row in df.iterrows():
            results["total"] += 1

            # row.to_dict()でdictに変換
            data = row.to_dict()

            validation = validate_sample(data, file_path)

            format_type = validation["format"]
            if format_type:
                results["by_format"][format_type]["total"] += 1

            if validation["valid"] is True:
                results["valid"] += 1
                if format_type:
                    results["by_format"][format_type]["valid"] += 1
            elif validation["valid"] is False:
                results["invalid"] += 1
                if format_type:
                    results["by_format"][format_type]["invalid"] += 1

                # エラーを記録（最初の10個まで）
                if len(results["errors"]) < 10:
                    results["errors"].append({
                        "index": idx,
                        "format": format_type,
                        "error": validation["error"],
                        "sample_id": validation.get("sample_id"),
                    })
            else:
                results["skipped"] += 1

        return results

    except ImportError:
        print(f"    Warning: pandas not available, skipping {file_path.name}")
        return None
    except Exception as e:
        print(f"    Warning: Error reading parquet: {e}")
        return None


def merge_results(result1, result2):
    """2つの結果をマージ"""
    if result1 is None:
        return result2
    if result2 is None:
        return result1

    merged = {
        "total": result1["total"] + result2["total"],
        "valid": result1["valid"] + result2["valid"],
        "invalid": result1["invalid"] + result2["invalid"],
        "skipped": result1["skipped"] + result2["skipped"],
        "by_format": defaultdict(lambda: {"total": 0, "valid": 0, "invalid": 0}),
        "errors": result1["errors"] + result2["errors"],
    }

    # フォーマット別の結果をマージ
    for fmt in set(list(result1["by_format"].keys()) + list(result2["by_format"].keys())):
        merged["by_format"][fmt]["total"] = result1["by_format"][fmt]["total"] + result2["by_format"][fmt]["total"]
        merged["by_format"][fmt]["valid"] = result1["by_format"][fmt]["valid"] + result2["by_format"][fmt]["valid"]
        merged["by_format"][fmt]["invalid"] = result1["by_format"][fmt]["invalid"] + result2["by_format"][fmt]["invalid"]

    return merged


def print_results(name, results):
    """検証結果を出力"""
    if results is None or results["total"] == 0:
        print(f"\n{name}: No data")
        return

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    valid_rate = results["valid"] / results["total"] * 100 if results["total"] > 0 else 0
    invalid_rate = results["invalid"] / results["total"] * 100 if results["total"] > 0 else 0

    print(f"Total samples:     {results['total']:,}")
    print(f"Valid:             {results['valid']:,} ({valid_rate:.2f}%)")
    print(f"Invalid:           {results['invalid']:,} ({invalid_rate:.2f}%)")
    print(f"Skipped:           {results['skipped']:,}")

    if results["by_format"]:
        print(f"\n{'─'*70}")
        print("Breakdown by format:")
        print(f"{'─'*70}")
        print(f"{'Format':<10} {'Total':>8} {'Valid':>8} {'Invalid':>8} {'Valid %':>10}")
        print(f"{'─'*70}")

        for fmt in sorted(results["by_format"].keys()):
            fmt_data = results["by_format"][fmt]
            fmt_valid_rate = fmt_data["valid"] / fmt_data["total"] * 100 if fmt_data["total"] > 0 else 0
            print(f"{fmt:<10} {fmt_data['total']:8,} {fmt_data['valid']:8,} {fmt_data['invalid']:8,} {fmt_valid_rate:9.2f}%")

    if results["errors"]:
        print(f"\n{'─'*70}")
        print(f"Sample errors (first {len(results['errors'])}):")
        print(f"{'─'*70}")

        for i, error in enumerate(results["errors"][:10], 1):
            location = error.get("line", error.get("index", "unknown"))
            print(f"\n{i}. Location: {location}")
            print(f"   Format: {error['format']}")
            print(f"   Sample ID: {error.get('sample_id', 'N/A')}")
            print(f"   Error: {error['error'][:200]}")


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

        print(f"\n{'#'*70}")
        print(f"Dataset: {dataset_dir.name}")
        print(f"{'#'*70}")

        dataset_results = None

        # JSONLファイルを処理
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            file_results = validate_jsonl_file(jsonl_file)
            dataset_results = merge_results(dataset_results, file_results)

        # Parquetファイルを処理
        for parquet_file in dataset_dir.glob("**/*.parquet"):
            file_results = validate_parquet_file(parquet_file)
            dataset_results = merge_results(dataset_results, file_results)

        if dataset_results and dataset_results["total"] > 0:
            all_results[dataset_dir.name] = dataset_results
            print_results(dataset_dir.name, dataset_results)

    # 全体統計
    if all_results:
        print(f"\n\n{'#'*70}")
        print("OVERALL VALIDATION RESULTS (All Datasets Combined)")
        print(f"{'#'*70}")

        overall_results = None
        for results in all_results.values():
            overall_results = merge_results(overall_results, results)

        print_results("All Datasets", overall_results)

        # データセット別サマリー
        print(f"\n{'─'*70}")
        print("Dataset Summary:")
        print(f"{'─'*70}")
        print(f"{'Dataset':<50} {'Total':>8} {'Valid':>8} {'Valid %':>10}")
        print(f"{'─'*70}")

        for name, results in sorted(all_results.items()):
            valid_rate = results["valid"] / results["total"] * 100 if results["total"] > 0 else 0
            print(f"{name:<50} {results['total']:8,} {results['valid']:8,} {valid_rate:9.2f}%")

        # 推奨事項
        print(f"\n\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}")

        overall_valid_rate = overall_results["valid"] / overall_results["total"] * 100 if overall_results["total"] > 0 else 0
        overall_invalid_rate = overall_results["invalid"] / overall_results["total"] * 100 if overall_results["total"] > 0 else 0

        print(f"\nOverall validation rate: {overall_valid_rate:.2f}%")
        print(f"Overall invalid rate: {overall_invalid_rate:.2f}%")

        if overall_valid_rate >= 99:
            print("\n✓✓ Excellent! Almost all samples are valid.")
        elif overall_valid_rate >= 95:
            print("\n✓ Good! Most samples are valid.")
        elif overall_valid_rate >= 90:
            print("\n△ Acceptable, but some samples have issues.")
        else:
            print("\n⚠ Warning! Many samples have validation errors.")

        if overall_results["invalid"] > 0:
            print(f"\n{overall_results['invalid']:,} samples failed validation.")
            print("Consider:")
            print("  - Reviewing the error samples")
            print("  - Filtering out invalid samples before training")
            print("  - Fixing the data generation process if this is synthetic data")


if __name__ == "__main__":
    main()
