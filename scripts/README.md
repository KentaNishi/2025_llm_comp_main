# Scripts

このディレクトリには、訓練環境と評価環境を管理するためのスクリプトが含まれています。

## 環境の概要

このプロジェクトでは2つの独立した仮想環境を使用します：

- **訓練環境 (.venv-train)**: モデルの訓練用
- **評価環境 (.venv-eval)**: StructEvalを使用したモデル評価用

これらは異なるバージョンのPyTorch等を使用するため、別々の仮想環境として分離されています。

## セットアップ

### 訓練環境のセットアップ

```bash
./scripts/setup_train.sh
```

### 評価環境のセットアップ

```bash
./scripts/setup_eval.sh
```

## 環境の切り替え

### 方法1: switch_env.shを使用（推奨）

```bash
# 訓練環境に切り替え
source scripts/switch_env.sh train

# 評価環境に切り替え
source scripts/switch_env.sh eval
```

### 方法2: 直接アクティベート

```bash
# 訓練環境
source .venv-train/bin/activate

# 評価環境
source .venv-eval/bin/activate
```

## 評価の実行

```bash
# スクリプト経由（自動で評価環境をアクティベート）
./scripts/run_eval.sh

# または、評価環境を手動でアクティベートしてから実行
source .venv-eval/bin/activate
cd StructEval
python3 -m structeval.cli --help
```

## 環境の確認

現在アクティブな環境を確認：

```bash
which python
# 出力例：
# /path/to/.venv-train/bin/python  # 訓練環境
# /path/to/.venv-eval/bin/python   # 評価環境
```

パッケージバージョンの確認：

```bash
uv pip list | grep torch
```

## トラブルシューティング

### 環境が混在している場合

```bash
# 仮想環境をディアクティベート
deactivate

# 必要に応じて環境を再作成
rm -rf .venv-train .venv-eval
./scripts/setup_train.sh
./scripts/setup_eval.sh
```

### flash-attnのインストールエラー

flash-attnは環境依存のため、インストールに失敗する場合があります。その場合は評価環境のセットアップスクリプトを編集し、該当部分をコメントアウトしてください。
