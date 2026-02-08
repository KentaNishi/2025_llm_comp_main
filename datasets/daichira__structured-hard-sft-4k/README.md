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
size_categories:
- 1K<n<10K
---

# Hard Synthetic Dataset for Structured Data Tasks (v1)

This dataset contains **4,000 high-difficulty synthetic samples** designed to improve LLM performance on complex structured data conversion, extraction, and formatting tasks.

The data is fully synthetic, generated using deterministic serialization to ensure syntax validity while maintaining high structural complexity (deep nesting and varied types).

## Dataset Summary

The dataset addresses four "hard" areas typically encountered in structured data processing:

| Task Subcategory | Task Type | Description | Count |
| :--- | :--- | :--- | :--- |
| **json_to_xml** | Transformation | Nested JSON to XML without attributes, preserving case. | 1,000 |
| **xml_to_yaml** | Transformation | Deeply nested XML to YAML, preserving tag names and structure. | 1,000 |
| **text_to_toml** | Extraction | Extracting attributes from text into TOML dotted tables/AOT. | 1,000 |
| **text_to_yaml** | Extraction | Extracting attributes from text into nested YAML structures. | 1,000 |

## Data Format

Each sample is in JSONL format with the following structure:
- `id`: A unique hash derived from the content.
- `category`: The high-level task category (e.g., `C_XML`, `C_TOML`, `C_YAML`).
- `subcategory`: Specific conversion/extraction type.
- `task`: Action (e.g., `transform`, `extract`).
- `seed`: Marker for generation source (`dummy_hard`).
- `messages`: Conversation format (User prompt and Assistant response).

## Features

- **Structural Complexity**: Objects are built with random depths, including nested arrays of dictionaries and mixed scalar types.
- **Strict Validity**: Every sample has been verified against standard parsers (PyYAML, tomllib/tomli, ElementTree).
- **Deterministic Serialization**: Assistant responses follow strict formatting rules (e.g., 2-space indentation, specific XML tag sanitization) to serve as a reliable ground truth.

## License

This dataset is licensed under **CC-BY-4.0** as it is a purely synthetic collection.

## Usage Note

This dataset is intended for supervised fine-tuning (SFT) of LLMs to improve their structural reasoning and formatting capabilities across various formats including XML, YAML, and TOML.
