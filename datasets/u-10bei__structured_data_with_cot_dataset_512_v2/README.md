
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
- chain-of-thought
- data_generation
pretty_name: Structured Data with CoT (512 tokens)
---
# structured_data_with_cot_dataset

このデータセットは、様々な形式（JSON、XML、YAML、TOML、CSV）の構造化データと、それぞれに対応する簡潔な思考連鎖（Chain-of-Thought, CoT）推論を含む多様な例を提供します。このデータセットの目的は、自然言語プロンプトに基づいて構造化データを生成できるモデルに対し、CoTを活用してより良い推論と出力品質を達成するための高品質な学習例を提供することです。

## データセットの概要

データセットの各エントリには、以下のものが含まれます。
- **`messages`**: OpenAIチャット形式に従ったメッセージ辞書のリストで、以下で構成されます。
  - `system`メッセージ: 特定のデータ形式の専門家としてのコンテキストを設定します。
  - `user`メッセージ: 特定のスキーマと形式で構造化データの生成を要求するプロンプトです。多様なプロンプトテンプレートが使用されています。
  - `assistant`メッセージ: 生成された構造化データに続く思考連鎖（CoT）推論が含まれます。
- **`metadata`**: 例に関する追加情報で、以下が含まれます。
  - `format`: 構造化データの形式（例: 'json'、'xml'、'yaml'、'toml'、'csv'）。
  - `complexity`: 生成されたスキーマの複雑度（'simple'、'medium'、'complex'）。より複雑な例が多く含まれるように調整されています。
  - `schema`: 表現される実世界のスキーマの種類（例: 'research_paper'、'clinical_note'、'api_specification'）。
  - `estimated_tokens`: アシスタントの応答のトークン数の推定値。

## サポートされるデータ形式

- **JSON** (JavaScript Object Notation)
- **XML** (Extensible Markup Language)
- **YAML** (YAML Ain't Markup Language)
- **TOML** (Tom's Obvious, Minimal Language) - 複雑なネストは簡素化される場合があります。
- **CSV** (Comma Separated Values) - 複数の行が生成され、フラット化された形式で表現されます。

## サポートされるスキーマ

このデータセットは、以下を含む幅広いコンパクトなスキーマをカバーしています。
- 研究論文 (例: `funding_source`のような条件付きフィールドを含む)
- 実験結果
- 診療記録
- 検査結果
- 処方箋
- 契約書
- 財務報告書
- 販売分析
- API仕様
- エラーログ
- 製品リスト (例: `is_available`のような条件付きフィールドを含む)
- 顧客レビュー
- 取引記録
- シラバス
- 学生評価
- ニュース記事
- ソーシャルメディア投稿

## 思考連鎖（Chain-of-Thought, CoT）推論

各`assistant`メッセージは、構造化データを生成するために取られたステップを概説する短いCoT推論ブロックから始まります。これにより、モデルは構造化データ生成に関わる論理と制約を理解し、正確で文脈に関連する出力を生成する能力を向上させることができます。CoTには、タスク、複雑度、形式ルール、構造、フィールドへのデータ入力に関する一般的なステップが含まれます。

## データセットの読み込み方法

このデータセットは`JSONL`（JSON Lines）形式で提供されています。Hugging Faceの`datasets`ライブラリを使用して簡単にロードできます。

```python
from datasets import load_dataset

# ローカルのJSONLファイルからデータセットをロード
dataset = load_dataset('json', data_files='structured_data_with_cot_dataset.jsonl', split='train')

# 例にアクセス
print(dataset[0])

# このデータセットをHugging Face Hubにプッシュすることもできます
# from huggingface_hub import notebook_login
# notebook_login() # ログインしていない場合
# dataset.push_to_hub("your-username/structured_data_with_cot_dataset")
```

## 生成方法

このデータセットは、`Faker`ライブラリを活用したPythonスクリプトを使用して、様々なコンパクトなスキーマに対して現実的な構造化データを生成しました。生成プロセスには、スキーマ、形式、複雑度をランダムに選択し、思考連鎖推論を埋め込むロジックが含まれており、大規模言語モデルのトレーニングに適した多様性と品質を保証します。特に、より多様なプロンプトテンプレートが使用され、データセットの複雑度の分布が調整され（より多くの「medium」および「complex」な例）、いくつかのスキーマジェネレーターには条件付きフィールド（例：研究論文の資金源、製品リストの在庫状況）や、CSVの動的な複数行生成ロジックが追加されています。TOML変換は、TOMLの仕様に合わせてネストされた構造を簡素化する場合があります。
