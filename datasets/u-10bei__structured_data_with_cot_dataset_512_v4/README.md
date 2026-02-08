
---
license: mit
tags:
- json
- xml
- yaml
- toml
- csv
- structured_data
- cot
pretty_name: Structured Data with CoT (512 tokens)
---
# structured_data_with_cot_dataset

このデータセットは、様々な形式（JSON、XML、YAML、TOML、CSV）の構造化データと、それぞれに対応する簡潔な思考連鎖（Chain-of-Thought, CoT）推論を含む多様な例を提供します。

## データセットの概要
- **messages**: OpenAIチャット形式 (system, user, assistant)
- **metadata**: format, complexity, schema, estimated_tokens

## サポートされるデータ形式
- JSON, XML, YAML, TOML, CSV

## 生成方法
Fakerライブラリを使用し、Pythonスクリプトで生成。検証用・テスト用に分割済み。