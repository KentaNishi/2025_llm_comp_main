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

# 3k Mixed Hard-Structured SFT Dataset (v1)

This dataset contains **3,000 synthetic samples** designed to improve LLM performance on complex structured data conversion, extraction, and formatting tasks.

It features a **perfectly balanced distribution** across 5 major data formats, ensuring comprehensive coverage of structural logic.

## Dataset Summary

The dataset consists of 3,000 samples, with an equal allocation of **20% (600 samples)** for each target format.

| Target Format | Count | Share | Description |
| :--- | :--- | :--- | :--- |
| **YAML** | 600 | 20% | Includes XML/Text/JSON/CSV/TOML to YAML conversions. |
| **TOML** | 600 | 20% | Focused on deep extraction from unstructured text to TOML. |
| **XML** | 600 | 20% | JSON/CSV to XML transformations. |
| **JSON** | 600 | 20% | Conversions from all other 5 sources to JSON. |
| **CSV** | 600 | 20% | Conversions from all other 5 sources to CSV. |

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
