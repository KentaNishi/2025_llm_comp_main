# Multi-Stage Training Guide

このガイドでは、拡張されたtrain.pyの使用方法を説明します。

## 機能

### 1. 段階的訓練（Multi-Stage Training）

複数のステージに分けて順次訓練を行うことができます。各ステージで前のステージの学習結果を引き継ぎます。

#### 使用方法

```bash
# Stage 1: 基礎訓練
python train.py --stage 1

# Stage 2: Stage 1の結果を引き継いで訓練
python train.py --stage 2

# Stage 3: Stage 2の結果を引き継いで訓練
python train.py --stage 3
```

#### 動作の仕組み

- **Stage 1**: ベースモデルから開始し、LoRAアダプターを訓練
  - 出力: `{OUT_LORA_DIR}_stage1/`

- **Stage 2以降**:
  1. 前のステージのアダプターをベースモデルにマージ
  2. マージ済みモデルに新しいLoRAアダプターを追加して訓練
  - 出力: `{OUT_LORA_DIR}_stage{N}/`

#### 設定ファイル

各ステージの設定は `stage{N}.env` ファイルで管理されます。

- `stage1.env`: 初期訓練の設定（高い学習率、短いステップ数）
- `stage2.env`: 中間訓練の設定（学習率を下げる、エポックベース）
- `stage3.env`: 最終調整の設定（低い学習率、特定タスクの強化）

### 2. データセットのMix機能

複数のデータセットを重み付きで混合して訓練できます。

#### 基本的な使用方法

##### 単一データセット（従来通り）

```bash
# stage1.envに設定
SFT_DATASET_ID=u-10bei/structured_data_with_cot_dataset_512_v2
```

##### 複数データセットのMix

```bash
# stage2.envに設定
SFT_DATASET_MIX='[
  {"id": "dataset1", "weight": 1.0, "split": "train"},
  {"id": "dataset2", "weight": 2.0, "split": "train"}
]'
```

#### パラメータ説明

- **id**: Hugging Face Hub上のデータセットID（必須）
- **weight**: サンプリングの重み（デフォルト: 1.0）
  - 大きいほど、そのデータセットから多くサンプリングされます
  - 例: weight=2.0のデータセットは、weight=1.0の2倍サンプリングされます
- **split**: データセットの分割（デフォルト: "train"）

#### 使用例

##### 例1: 2つのデータセットを等しく混合

```bash
SFT_DATASET_MIX='[
  {"id": "dataset_A", "weight": 1.0},
  {"id": "dataset_B", "weight": 1.0}
]'
```

##### 例2: 特定のデータセットを重視

```bash
SFT_DATASET_MIX='[
  {"id": "general_data", "weight": 1.0},
  {"id": "toml_specialized_data", "weight": 3.0}
]'
```
→ TOML特化データセットから3倍多くサンプリング

##### 例3: 3つ以上のデータセット

```bash
SFT_DATASET_MIX='[
  {"id": "dataset_base", "weight": 2.0},
  {"id": "dataset_json", "weight": 1.5},
  {"id": "dataset_toml", "weight": 2.5},
  {"id": "dataset_yaml", "weight": 1.0}
]'
```

#### サンプリングの仕組み

1. 各データセットをロード
2. 重みを正規化（合計=1.0になるように）
3. 最大データセットサイズを基準に、各データセットから重みに応じてサンプリング
4. すべてをシャッフルして混合

**例:**
- Dataset A: 1000サンプル, weight=1.0
- Dataset B: 500サンプル, weight=2.0

→ 正規化後の重み: A=0.33, B=0.67
→ 基準サイズ: 1000（最大）
→ サンプリング数: A=330, B=670
→ 最終データセット: 1000サンプル（シャッフル済み）

## 完全な使用例

### Stage 1: 基礎訓練

**stage1.env:**
```bash
SFT_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SFT_DATASET_ID=u-10bei/structured_data_with_cot_dataset_512_v2
SFT_OUT_LORA_DIR=/content/lora_structeval_t_qwen3_4b

SFT_MAX_SEQ_LEN=2048
SFT_LORA_R=8
SFT_LORA_ALPHA=32
SFT_EPOCHS=1
SFT_MAX_STEPS=1500
SFT_LR=1e-4
```

**実行:**
```bash
python train.py --stage 1
```

### Stage 2: 複数データセットでの訓練

**stage2.env:**
```bash
SFT_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SFT_OUT_LORA_DIR=/content/lora_structeval_t_qwen3_4b

# データセットをMix
SFT_DATASET_MIX='[
  {"id": "u-10bei/structured_data_with_cot_dataset_512_v2", "weight": 1.0},
  {"id": "additional_toml_dataset", "weight": 2.0}
]'

SFT_MAX_SEQ_LEN=2048
SFT_LORA_R=8
SFT_EPOCHS=1
SFT_LR=5e-5

# アップサンプリングも併用可能
SFT_USE_UPSAMPLING=1
SFT_UPSAMPLE_RULES='{"text_to_toml": 3.0, "toml_to_yaml": 2.0}'
```

**実行:**
```bash
python train.py --stage 2
```

### Stage 3: 最終調整

**stage3.env:**
```bash
SFT_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SFT_OUT_LORA_DIR=/content/lora_structeval_t_qwen3_4b

SFT_DATASET_MIX='[
  {"id": "dataset_balanced", "weight": 1.0},
  {"id": "dataset_hard_cases", "weight": 1.5}
]'

SFT_EPOCHS=1
SFT_LR=2e-5
```

**実行:**
```bash
python train.py --stage 3
```

## 出力ディレクトリ構造

```
/content/lora_structeval_t_qwen3_4b_stage1/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── ...

/content/lora_structeval_t_qwen3_4b_stage2_merged_base/  # 内部使用（マージ済みモデル）
  └── ...

/content/lora_structeval_t_qwen3_4b_stage2/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── ...

/content/lora_structeval_t_qwen3_4b_stage3/
  ├── adapter_config.json
  ├── adapter_model.safetensors
  └── ...
```

## MLflowによる実験管理

訓練の進捗とメトリクスは自動的にMLflowに記録されます。

### 記録される情報

- 学習率、損失、評価スコアの推移
- すべてのハイパーパラメータ（学習率、バッチサイズ、LoRA設定など）
- 各ステージの学習履歴
- システム情報（GPU使用率、メモリなど）

### MLflow UIの起動

訓練データはリポジトリ内の `mlruns/` ディレクトリに保存されます。

**方法1: 起動スクリプトを使用（推奨）**

```bash
./official_content/mlflow_ui.sh
```

**方法2: 直接コマンドを実行**

```bash
mlflow ui --backend-store-uri file:///home/nkutm/workspace/2025-llm-advance-competition-main/mlruns --port 58000
```

ブラウザで http://localhost:58000 にアクセスすると、学習履歴やメトリクスを可視化できます。

### 実験名のカスタマイズ

各ステージの `.env` ファイルで実験名を変更できます。

```bash
# stage1.envに追加
MLFLOW_EXPERIMENT_NAME=my-custom-experiment-stage1
```

デフォルトでは `llm-training-stage{N}` という実験名が使用されます。

## 注意事項

1. **メモリ使用量**: Stage 2以降では、アダプターのマージ時に一時的にメモリ使用量が増加します
2. **データセットMix**: `SFT_DATASET_MIX`が設定されている場合、`SFT_DATASET_ID`は無視されます
3. **後方互換性**: `SFT_DATASET_MIX`を設定しなければ、従来通り単一データセットで動作します
4. **アップサンプリング**: データセットMixとアップサンプリング（`SFT_USE_UPSAMPLING`）は併用可能です

## トラブルシューティング

### 前のステージのアダプターが見つからない

```
[WARNING] Previous stage adapter not found at /path/to/adapter_stage1, starting from base model
```

→ 前のステージを実行していない、または出力ディレクトリが異なる場合に表示されます。
   問題なければ、ベースモデルから開始されます。

### データセットのロード失敗

```
[ERROR] Failed to load dataset_name: ...
```

→ データセットIDが正しいか、ネットワーク接続を確認してください。
   他のデータセットがロードされていれば、訓練は継続されます。

### JSON形式のエラー

```
[WARNING] Failed to parse DATASET_MIX: ...
```

→ JSON形式が正しいか確認してください。シングルクォートで囲み、内部はダブルクォートを使用します。
