# ============================================================
# 3) LoRAアダプターをHugging Faceへアップロード (作成済みのREADMEを含む)
# ============================================================

import os
import fnmatch
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# execution.env から環境変数を読み込む
load_dotenv("execution.env")

# 環境変数取得用のヘルパー関数
def _getenv(key: str, default: str = "") -> str:
    """環境変数を取得する。存在しない場合はデフォルト値を返す"""
    return os.environ.get(key, default)

# Hugging Face APIの操作用インスタンスを作成
api = HfApi()

# 各種パスや設定の準備
LORA_SAVE_DIR = Path(_getenv("SFT_OUT_LORA_DIR", "/content/lora_output"))  # 学習済みモデルが保存されているディレクトリ
HF_REPO_ID    = _getenv("HF_REPO_ID", "your_id/your-lora-repo")  # アップロード先のレポジトリID

# 非公開設定の確認（環境変数が '1' または 'true' ならプライベート設定にする）
PRIVATE       = _getenv("HF_PRIVATE", "1") in ("1","true","True")

# -----------------------------
# 3.1) 必須ファイルの存在確認
# -----------------------------
# アップロードに最低限必要なファイルを定義します
required_files = {
    "adapter_config.json", # LoRAの設定ファイル
    "README.md",           # 受講生が作成した解説文書
}

# 保存ディレクトリにあるファイル名のリストを取得
present = {p.name for p in LORA_SAVE_DIR.iterdir() if p.is_file()}

# 足りないファイルをリストアップ
missing = [f for f in required_files if f not in present]

# モデル本体（adapter_model.safetensors または .bin）が存在するか確認
if not any(f.startswith("adapter_model.") for f in present):
    missing.append("adapter_model.(safetensors|bin)")

# 必須ファイルが欠けている場合は、エラーを表示して処理を中断します
if missing:
    raise RuntimeError(
        "アップロードを中止しました。\n"
        "以下の必須ファイルが見つかりません:\n"
        + "\n".join(f"- {m}" for m in missing) +
        "\n\nアップロード前に、README.md を手書きで作成し保存してください。"
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
STAGE_DIR = Path("/content/hf_upload_stage")

if STAGE_DIR.exists():
    shutil.rmtree(STAGE_DIR) # 既存のフォルダがあれば一旦削除
STAGE_DIR.mkdir(parents=True)

# 許可されたファイルだけを一時フォルダにコピー
for p in LORA_SAVE_DIR.iterdir():
    if p.is_file() and is_allowed(p.name):
        (STAGE_DIR / p.name).write_bytes(p.read_bytes())

print("📦 アップロード対象ファイル:", [p.name for p in STAGE_DIR.iterdir()])

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