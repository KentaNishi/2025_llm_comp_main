# ============================================================
# 3) LoRAアダプターをHugging Faceへアップロード (作成済みのREADMEを含む)
# ============================================================
#
# 使用方法:
#   python upload_model_to_hf.py <adapter_dir>
#   例: python upload_model_to_hf.py ../lora_output/lora_structeval_t_qwen3_4b
#       python upload_model_to_hf.py /content/lora_output
#
# HF_REPO_ID は DLNorb/<adapter_dir のディレクトリ名> として自動設定されます。
# adapter_config.json が直下にない場合は checkpoint-\d+ サブディレクトリを探します。

import os
import re
import sys
import fnmatch
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# スクリプトと同じディレクトリの env ファイルを読み込む（実行ディレクトリに依存しない）
_HERE = Path(__file__).parent
load_dotenv(_HERE / "execution.env")


# 環境変数取得用のヘルパー関数
def _getenv(key: str, default: str = "") -> str:
    """環境変数を取得する。存在しない場合はデフォルト値を返す"""
    return os.environ.get(key, default)


def resolve_adapter_dir(base_dir: Path) -> Path:
    """adapter_config.json があるディレクトリを返す。
    直下になければ checkpoint-\\d+ サブディレクトリの中で最新のものを探す。"""
    if (base_dir / "adapter_config.json").exists():
        return base_dir
    checkpoints = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and re.match(r"^checkpoint-\d+$", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    if checkpoints:
        latest = checkpoints[-1]
        print(f"[INFO] adapter_config.json not found in root, using: {latest}")
        return latest
    return base_dir


# Hugging Face APIの操作用インスタンスを作成
api = HfApi()

# ------------------------------------------------------------------
# コマンドライン引数からアダプターディレクトリを取得
# ------------------------------------------------------------------
if len(sys.argv) < 2:
    # 引数なしの場合は環境変数から取得
    lora_dir_str = _getenv("SFT_OUT_LORA_DIR", "/content/lora_output")
else:
    lora_dir_str = sys.argv[1]

LORA_BASE_DIR = Path(lora_dir_str).expanduser().resolve()

# checkpoint サブディレクトリを解決
LORA_SAVE_DIR = resolve_adapter_dir(LORA_BASE_DIR)

# ディレクトリ名からモデル名を抽出し、HF repo IDを構築
# （checkpoint サブディレクトリの場合でも親ディレクトリ名を使う）
model_name = LORA_BASE_DIR.name
HF_REPO_ID = f"DLNorb/{model_name}"

print(f"[INFO] Adapter base dir  : {LORA_BASE_DIR}")
print(f"[INFO] Adapter files from: {LORA_SAVE_DIR}")
print(f"[INFO] Model name        : {model_name}")
print(f"[INFO] HF Repo ID        : {HF_REPO_ID}")

# 非公開設定の確認（環境変数が '1' または 'true' ならプライベート設定にする）
PRIVATE = _getenv("HF_PRIVATE", "0") in ("1", "true", "True")

# -----------------------------
# 3.1) 必須ファイルの存在確認
# -----------------------------
# アップロードに最低限必要なファイルを定義します
required_files = {
    "adapter_config.json",  # LoRAの設定ファイル（checkpoint dir から）
}
readme_required = {"README.md"}  # README は親ディレクトリから

# adapter ファイルが存在するディレクトリのファイル一覧
present_adapter = {p.name for p in LORA_SAVE_DIR.iterdir() if p.is_file()}
# README は親ディレクトリから探す
present_base = {p.name for p in LORA_BASE_DIR.iterdir() if p.is_file()}

missing = [f for f in required_files if f not in present_adapter]
missing += [f for f in readme_required if f not in present_base]

# モデル本体（adapter_model.safetensors または .bin）が存在するか確認
if not any(f.startswith("adapter_model.") for f in present_adapter):
    missing.append("adapter_model.(safetensors|bin)")

# 必須ファイルが欠けている場合は、エラーを表示して処理を中断します
if missing:
    raise RuntimeError(
        "アップロードを中止しました。\n"
        "以下の必須ファイルが見つかりません:\n"
        + "\n".join(f"- {m}" for m in missing)
        + "\n\nアップロード前に、README.md を generate_README.py で生成してください。"
    )

print("✅ 必須ファイルの確認が完了しました。")

# -----------------------------
# 3.2) アップロード対象の選別（ホワイトリスト）
# -----------------------------
# 不要な一時ファイルなどをアップロードしないよう、許可するファイル形式を指定します
ALLOW_PATTERNS = [
    "README.md",
    "adapter_config.json",
    "adapter_model.*",
    "tokenizer.*",
    "special_tokens_map.json",
    "*.json",
]


def is_allowed(name: str) -> bool:
    """ファイル名が許可パターンに一致するか判定する関数"""
    return any(fnmatch.fnmatch(name, pat) for pat in ALLOW_PATTERNS)


# アップロード用の一時フォルダ（ステージング領域）を作成
STAGE_DIR = Path("/tmp/hf_upload_stage")

if STAGE_DIR.exists():
    shutil.rmtree(STAGE_DIR)  # 既存のフォルダがあれば一旦削除
STAGE_DIR.mkdir(parents=True)

# adapter ファイルを一時フォルダにコピー（checkpoint dir or base dir）
for p in LORA_SAVE_DIR.iterdir():
    if p.is_file() and is_allowed(p.name):
        (STAGE_DIR / p.name).write_bytes(p.read_bytes())

# README は親ディレクトリから取得（checkpoint dir にない場合も対応）
readme_src = LORA_BASE_DIR / "README.md"
if readme_src.exists():
    (STAGE_DIR / "README.md").write_bytes(readme_src.read_bytes())

print("📦 アップロード対象ファイル:", sorted(p.name for p in STAGE_DIR.iterdir()))

# -----------------------------
# 3.3) リポジトリ作成とアップロード
# -----------------------------

# Hugging Face上にリポジトリを作成（既に存在していてもOK）
api.create_repo(
    repo_id=HF_REPO_ID,
    repo_type="model",
    exist_ok=True,
    private=PRIVATE,
)

# 一時フォルダの内容をまるごとアップロード
api.upload_folder(
    folder_path=str(STAGE_DIR),
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Upload LoRA adapter (README written by author)",
)

print("✅ アップロードが正常に完了しました。")
print(f"URL: https://huggingface.co/{HF_REPO_ID}")
