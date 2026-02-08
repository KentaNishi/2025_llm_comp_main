# 評価環境構築と実行 - 2026-02-07

## 作業概要

evaluation.pyを実行するための評価環境を構築し、run_eval.shを通じて正常に実行できるようにした。

## 環境情報

- OS: WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- Python: 3.12.3
- CUDA: 12.8
- 使用モデル: Qwen/Qwen3-4B-Instruct-2507 (ベースモデル)

## 発生した問題と解決策

### 1. fireパッケージのバージョン不一致

**問題:**
```
Because only fire<=0.7.1 is available and
2025_llm_comp_main:eval depends on fire>=1.0.0
```

**原因:**
- pyproject.tomlで`fire>=1.0.0`を指定していたが、実際には0.7.1が最新版

**解決策:**
```toml
# pyproject.toml
fire>=0.7.0  # >=1.0.0 から変更
```

### 2. flash-attnのビルドエラー

**問題:**
```
ModuleNotFoundError: No module named 'torch'
hint: `flash-attn` depends on `torch`, but doesn't declare it as a build dependency
```

**原因:**
- flash-attnをソースからビルドする際にPyTorchが必要だが、ビルド時に利用できない
- pyproject.tomlに含めると自動ビルドが試みられる

**解決策:**
1. pyproject.tomlからflash-attnを削除
2. ビルド済みwheelファイルを直接インストール

```bash
# setup_eval.sh
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
uv pip install "$FLASH_ATTN_WHEEL"
```

**選定基準:**
- Python 3.12 (cp312)
- PyTorch 2.9 (torch2.9)
- CUDA 12.x (cu12)
- CXX11 ABI TRUE (cxx11abiTRUE)
- Linux x86_64アーキテクチャ

### 3. 複数仮想環境でのuv sync警告

**問題:**
```
warning: `VIRTUAL_ENV=.venv-eval` does not match the project environment path `.venv`
```

**原因:**
- プロジェクトに複数の仮想環境(.venv, .venv-train, .venv-eval)が存在
- uv syncがデフォルトで.venvを期待

**解決策:**
```bash
# setup_train.sh と setup_eval.sh
uv sync --active --no-group eval  # または --only-group eval
```

`--active`オプションで現在アクティブな仮想環境を対象にする。

### 4. PyTorchバージョンの上書き

**問題:**
- StructEvalのインストール後、PyTorch 2.10.0がインストールされる
- 評価環境ではPyTorch 2.9.0が必要

**原因:**
- StructEvalのsetup.pyで`torch`（バージョン指定なし）を依存関係に含んでいる
- 最新版（2.10.0）がインストールされる

**対処:**
手動で再インストール:
```bash
source .venv-eval/bin/activate
uv pip uninstall flash-attn
uv pip install "https://github.com/.../flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### 5. vLLMのマルチプロセス起動エラー

**問題:**
```
An attempt has been made to start a new process before the current process
has finished its bootstrapping phase.
```

**原因:**
- vLLMがワーカープロセスを`spawn`モードで起動
- Pythonスクリプトに`if __name__ == "__main__":`ガードがない

**解決策:**
evaluation.pyを修正:
```python
if __name__ == "__main__":
    # メイン処理をここに移動
    print("[INFO] Try configs (in order):")
    # ...
    for idx, cfg in enumerate(TRY_CONFIGS, 1):
        # ...
```

### 6. ファイルパスの修正

**問題:**
- Google Colab用のパス(`/content/`)がハードコードされていた

**解決策:**
```python
# evaluation.py
INPUT_PATH  = "/root/workspace/2025_llm_comp_main/official_content/public_150.json"
OUTPUT_PATH = "/root/workspace/2025_llm_comp_main/outputs/inference.json"
```

## セットアップ手順

### 1. 評価環境の構築

```bash
# 評価環境のセットアップ
./scripts/setup_eval.sh
```

実行内容:
1. .venv-eval仮想環境の作成
2. StructEvalリポジトリのクローン
3. 評価用依存パッケージのインストール（PyTorch 2.9.0、vLLM 0.13.0など）
4. flash-attn 2.8.1のビルド済みwheelインストール
5. StructEvalのインストール
6. outputsディレクトリの作成

### 2. 評価の実行

```bash
# 評価スクリプトの実行
./scripts/run_eval.sh
```

実行内容:
1. .venv-eval環境のアクティベート
2. evaluation.pyの実行
3. 150問の推論実行
4. outputs/inference.jsonへの出力

## 実行結果

### 成功メトリクス

- ✅ 150問すべての推論が完了
- ✅ 出力ファイル: `outputs/inference.json` (161KB)
- ✅ 処理時間: 約14分
  - モデル読み込み: 約11分（初回ダウンロード含む）
  - 推論処理: 約2分19秒
- ✅ 推論速度:
  - 入力: 369.05 tokens/s
  - 出力: 297.70 tokens/s

### 設定パラメータ

使用された設定（1回目の試行で成功）:
```python
max_model_len = 4096
max_tokens = 4096
gpu_memory_utilization = 0.8
temperature = 0.0
enforce_eager = True
```

GPU KV cache: 26,784 tokens
最大並行度: 6.54x (4,096トークンのリクエスト)

## 重要な知見

### 1. WSL環境での注意点

- `pin_memory=False`が自動設定される（パフォーマンスへの影響あり）
- CUDA 12.8環境でPyTorch 2.9.0とflash-attn 2.8.1が正常動作

### 2. flash-attnの選定方法

GitHub ReleasesからAPIで取得:
```bash
curl -s https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.1
```

確認すべき項目:
- Pythonバージョン (cp312 = Python 3.12)
- PyTorchバージョン (torch2.9)
- CUDAバージョン (cu12)
- CXX11 ABI (True/False) - `torch._C._GLIBCXX_USE_CXX11_ABI`で確認
- アーキテクチャ (x86_64/aarch64)

### 3. uv syncの仮想環境管理

複数の仮想環境を使い分ける場合:
```bash
# 訓練環境
source .venv-train/bin/activate
uv sync --active --no-group eval

# 評価環境
source .venv-eval/bin/activate
uv sync --active --only-group eval
```

`--active`オプションが必須。

### 4. vLLMの安定実行

環境変数の設定（evaluation.py内で実施済み）:
```python
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
```

メイン処理の保護:
```python
if __name__ == "__main__":
    # すべてのLLM実行コードをここに
```

### 5. 依存パッケージの競合解決

StructEvalがtorchをバージョン指定なしで依存:
```python
# StructEval/setup.py
install_requires=[
    "torch",  # バージョン指定なし → 最新版がインストールされる
    ...
]
```

対策:
- インストール後に必要なバージョンを強制再インストール
- または、setup.pyを編集してバージョンを固定

## ファイル構成

```
.
├── scripts/
│   ├── setup_train.sh    # 訓練環境セットアップ（修正済み）
│   ├── setup_eval.sh     # 評価環境セットアップ（修正済み）
│   ├── run_eval.sh       # 評価実行スクリプト（修正済み）
│   └── README.md         # スクリプト説明
├── official_content/
│   ├── evaluation.py     # 評価スクリプト（修正済み）
│   └── public_150.json   # 評価データ
├── outputs/
│   └── inference.json    # 推論結果
├── pyproject.toml        # 依存関係定義（修正済み）
└── StructEval/           # 評価フレームワーク（クローン済み）
```

## 今後の改善案

### 1. StructEvalの依存関係固定

StructEval/setup.pyを編集してPyTorchバージョンを固定:
```python
install_requires=[
    "torch>=2.9.0,<2.10.0",
    ...
]
```

### 2. setup_eval.shの改善

PyTorchの再インストール処理を追加:
```bash
# StructEvalインストール後にPyTorchを再インストール
echo "Re-installing PyTorch 2.9.0..."
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 triton==3.5.0 --force-reinstall
```

### 3. モデルキャッシュの活用

初回実行でモデルダウンロードに時間がかかるため、事前ダウンロードを推奨:
```python
from transformers import AutoModel
AutoModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
```

### 4. GPU設定の最適化

より多くのGPUメモリを使用可能な場合:
```python
gpu_memory_utilization = 0.9  # 0.8から増やす
```

### 5. 並列実行の検討

複数GPUがある場合、tensor_parallel_sizeを増やして高速化:
```python
tensor_parallel_size = 2  # GPUの数に応じて調整
```

## チェックリスト

環境構築時の確認項目:

- [ ] Python 3.12がインストールされている
- [ ] CUDA 12.xがインストールされている
- [ ] uvパッケージマネージャーがインストールされている
- [ ] pyproject.tomlでfire>=0.7.0に修正済み
- [ ] setup_train.shとsetup_eval.shに--activeオプション追加済み
- [ ] flash-attnのビルド済みwheelをダウンロード
- [ ] evaluation.pyのパスを環境に合わせて修正
- [ ] evaluation.pyに`if __name__ == "__main__":`ガード追加
- [ ] .venv-eval環境でPyTorch 2.9.0を確認
- [ ] outputsディレクトリが存在する

実行前の確認項目:

- [ ] 現在の仮想環境を確認 (`which python3`)
- [ ] PyTorchバージョンを確認 (`python3 -c "import torch; print(torch.__version__)"`)
- [ ] CUDAが利用可能か確認 (`python3 -c "import torch; print(torch.cuda.is_available())"`)
- [ ] flash-attnがインストールされているか確認 (`uv pip list | grep flash-attn`)
- [ ] 入力ファイルが存在するか確認 (`ls official_content/public_150.json`)

## 参考リンク

- flash-attention releases: https://github.com/Dao-AILab/flash-attention/releases
- StructEval repository: https://github.com/Osakana7777777/StructEval
- vLLM documentation: https://docs.vllm.ai/
- uv documentation: https://docs.astral.sh/uv/

## まとめ

WSL環境でvLLMを使用した評価環境の構築は、依存関係の複雑さ（特にPyTorchとflash-attnの組み合わせ）により課題が多いが、ビルド済みwheelファイルの使用と適切な環境分離により安定動作を実現できた。

重要なポイント:
1. **ビルド済みwheelの使用** - flash-attnはソースビルドを避ける
2. **仮想環境の明示的管理** - uvに--activeオプションを使用
3. **マルチプロセス対応** - vLLM使用時は必ず__main__ガードを追加
4. **バージョン固定** - PyTorchのバージョンが上書きされないよう注意

これらの知見により、`./scripts/run_eval.sh`を実行するだけで150問の評価が安定して実行できる環境が整った。
