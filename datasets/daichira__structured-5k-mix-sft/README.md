---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
tags:
- structured-data
- sft
- synthetic
- json
- xml
- yaml
- toml
- csv
size_categories:
- 1K<n<10K
---

# 5k Mixed Hard-Structured SFT Dataset (v1)

This dataset contains **5,000 synthetic samples** designed to improve LLM performance on complex structured data conversion, extraction, and formatting tasks.

It aggregates 13 distinct conversion tasks with a specific focus on format diversity and structural complexity.

## Dataset Summary

The dataset is distributed across five major formats with the following allocation:

| Target Format | Count | Share | Task Types |
| :--- | :--- | :--- | :--- |
| **YAML** | 1,500 | 30% | `xml_to_yaml`, `text_to_yaml`, `json_to_yaml`, `csv_to_yaml`, `toml_to_yaml` |
| **TOML** | 1,500 | 30% | `text_to_toml` (Extraction focus) |
| **XML** | 1,000 | 20% | `json_to_xml`, `csv_to_xml` |
| **JSON** | 500 | 10% | `text_to_json`, `csv_to_json`, `xml_to_json`, `yaml_to_json`, `toml_to_json` |
| **CSV** | 500 | 10% | `text_to_csv`, `json_to_csv`, `xml_to_csv`, `yaml_to_csv` |

## Data Format

Each sample is in JSONL format with the following structure:
- `id`: A unique hash derived from the content.
- `category`: The high-level task category (e.g., `C_XML`, `C_TOML`).
- `subcategory`: Specific conversion/extraction type (e.g., `xml_to_yaml`).
- `task`: Action (`transform` or `extract`).
- `seed`: Marker for generation source (`dummy_hard` or `bench_fill`).
- `messages`: Conversation format (User prompt and Assistant response).

## Features

- **Strict XML Structure**: XML outputs use flattened list representations (repeating tags) rather than generic `<item>` wrappers, ensuring better semantic alignment.
- **Clean TOML/YAML Extraction**: Text-to-Structured tasks output **only** the requested attributes as flat key-value pairs or dotted tables, avoiding unnecessary root wrappers.
- **Deterministic Validity**: All samples are strictly validated against standard parsers (PyYAML, tomllib, ElementTree, csv).

## License

This dataset is licensed under **CC-BY-4.0** as it is a purely synthetic collection.

## Citation

If you use this dataset in your research, please cite the StructEval-T project.

---
*Note: This dataset is intended for supervised fine-tuning (SFT) of LLM agents to improve their structural reasoning and formatting capabilities.*
