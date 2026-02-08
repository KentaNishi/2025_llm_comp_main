# 訓練環境セットアップ - 2026-02-08

## 作業概要

`train.py` を実行するための訓練環境 (`.venv-train`) の初期セットアップ手順と、発生した問題・解決策をまとめる。

## 前提条件

- OS: Linux (Ubuntu)
- Python: 3.12（システムPython `/usr/bin/python3.12`）
- パッケージ管理: uv
- GPU: NVIDIA GPU (CUDA対応)

## 初期セットアップ手順

### 1. 訓練用仮想環境の構築

```bash
cd /root/workspace/2025_llm_comp_main
./scripts/setup_train.sh
```

内部で行われること:
1. `.venv-train` 仮想環境の作成 (`uv venv .venv-train`)
2. 訓練用依存パッケージのインストール (`uv sync --active --no-group eval`)

### 2. python3-dev パッケージのインストール（必須）

Triton がランタイムで C コードをコンパイルする際に `Python.h` ヘッダーが必要。
`.venv-train` がシステムPython (`/usr/bin/python3.12`) を使用しているため、apt で開発ヘッダーを入れる必要がある。

```bash
sudo apt-get install -y python3.12-dev
```

**補足**: uv 管理のPython (`uv python install 3.12`) を使っている場合はヘッダーが同梱されるため、この手順は不要。

**確認方法:**
```bash
ls /usr/include/python3.12/Python.h
```

**この手順を怠った場合のエラー:**
```
/tmp/xxx/cuda_utils.c:6:10: fatal error: Python.h: No such file or directory
```

### 3. 訓練の実行

```bash
.venv-train/bin/python ./official_content/train.py --stage 1
```

## 環境構成

### 仮想環境の構造

```
.venv-train/
├── bin/python → /usr/bin/python3.12  (システムPython)
└── pyvenv.cfg
    ├── home = /usr/bin
    ├── uv = 0.10.0
    └── version_info = 3.12.3
```

### 主要パッケージバージョン (pyproject.toml)

| パッケージ | バージョン | 用途 |
|---|---|---|
| torch | 2.9.0+cu128 | PyTorch (CUDA 12.8ビルド) |
| transformers | 4.56.2 | Hugging Face Transformers |
| unsloth | 2025.12.7 | 高速ファインチューニング |
| unsloth-zoo | 2025.12.7 | Unsloth拡張 |
| datasets | 4.3.0 | データセット管理 |
| peft | 0.13.2 | LoRAアダプタ |
| accelerate | 1.4.0 | 分散学習 |
| bitsandbytes | >=0.45.0 | 量子化 |
| mlflow | >=3.9.0 | 実験管理 |
| trl | 0.24.0 | Transformer強化学習 |
| triton | 3.5.0 | GPU カーネルコンパイラ (torch同梱) |

### 訓練設定ファイル (stage1.env)

```
SFT_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SFT_DATASET_ID=daichira/structured-5k-mix-sft
SFT_MAX_SEQ_LEN=2048
SFT_LORA_R=8
SFT_LORA_ALPHA=32
SFT_EPOCHS=1
SFT_MAX_STEPS=1500
SFT_PER_DEVICE_TRAIN_BS=1
SFT_GRAD_ACCUM=16
SFT_LR=1e-4
```

## 発生した問題と解決策

### 1. Python.h が見つからない

**エラー:**
```
/tmp/xxx/cuda_utils.c:6:10: fatal error: Python.h: No such file or directory
```

**原因:**
- Triton が CUDA ユーティリティをコンパイルする際に Python 開発ヘッダーが必要
- `.venv-train` がシステムPython を使用しており、`python3.12-dev` パッケージ未インストール

**解決策:**
```bash
apt-get install -y python3.12-dev
```

**uv との関係:**
- `python3.12-dev` は C ヘッダーファイルを追加するだけで、Python バイナリやライブラリは変更しない
- uv が管理する仮想環境やパッケージとは管轄が異なるため衝突しない

### 2. GPU アーキテクチャと CUDA Toolkit の互換性（ハードウェア依存）

**エラー:**
```
ptxas fatal : Value 'sm_103a' is not defined for option 'gpu-name'
```

**原因:**
- NVIDIA B300 (Blackwell Ultra) の Compute Capability 10.3 (`sm_103a`) が CUDA Toolkit 12.8 の `ptxas` に未対応

**GPU別の CUDA Toolkit 互換性:**

| GPU | Compute Capability | sm | CUDA 12.8 | CUDA 13.0+ |
|---|---|---|---|---|
| B200 | 10.0 | sm_100a | 対応 | 対応 |
| B300 | 10.3 | sm_103a | 非対応 | 必要 |
| RTX 5090/5080 | 12.0 | sm_120a | 対応 | 対応 |

**対処:**
- B200 インスタンスに切り替えれば CUDA 12.8 のまま動作可能
- B300 を使う場合は CUDA Toolkit 13.0+ のインストールが必要

## セットアップチェックリスト

新しいインスタンスでの作業開始時に確認する項目:

### 環境構築

- [ ] リポジトリをクローン / 同期
- [ ] `python3.12-dev` をインストール (`apt-get install -y python3.12-dev`)
- [ ] `./scripts/setup_train.sh` を実行
- [ ] `./scripts/setup_eval.sh` を実行

### 動作確認

- [ ] Python バージョン確認: `.venv-train/bin/python --version` → 3.12.x
- [ ] CUDA 利用可能: `.venv-train/bin/python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] GPU アーキテクチャ確認: `.venv-train/bin/python -c "import torch; print(torch.cuda.get_device_capability())"`
  - (10, 0) = B200 → CUDA 12.8 OK
  - (10, 3) = B300 → CUDA 13.0+ 必要
- [ ] 主要モジュールのインポート確認:
  ```bash
  .venv-train/bin/python -c "
  for m in ['dotenv','numpy','torch','datasets','transformers','unsloth','mlflow','peft']:
      mod=__import__(m); print(f'{m}: {getattr(mod,\"__version__\",\"ok\")}')
  "
  ```
- [ ] Python.h 存在確認: `ls /usr/include/python3.12/Python.h`

### 訓練実行

- [ ] `stage1.env` の設定内容を確認
- [ ] `.venv-train/bin/python ./official_content/train.py --stage 1` で訓練開始

## ファイル構成

```
.
├── scripts/
│   ├── setup_train.sh       # 訓練環境セットアップ
│   ├── setup_eval.sh        # 評価環境セットアップ
│   ├── run_eval.sh          # 評価実行スクリプト
│   └── README.md            # スクリプト説明
├── official_content/
│   ├── train.py             # 訓練スクリプト
│   ├── evaluation.py        # 評価スクリプト
│   ├── stage1.env           # Stage 1 訓練パラメータ
│   ├── stage2.env           # Stage 2 訓練パラメータ
│   └── stage3.env           # Stage 3 訓練パラメータ
├── .venv-train/             # 訓練用仮想環境
├── .venv-eval/              # 評価用仮想環境
└── pyproject.toml           # 依存関係定義
```

## 参考

- uv ドキュメント: https://docs.astral.sh/uv/
- Unsloth: https://github.com/unslothai/unsloth
- CUDA GPU Compute Capability: https://developer.nvidia.com/cuda/gpus
