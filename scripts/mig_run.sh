#!/bin/bash
# MIG 空きインスタンス自動選択スクリプト
#
# 使い方:
#   ./scripts/mig_run.sh [--venv <path>] <コマンド> [引数...]
#
# 例:
#   ./scripts/mig_run.sh --venv .venv-eval python official_content/evaluation.py
#   ./scripts/mig_run.sh --venv .venv-train python official_content/train.py --stage 1
#   ./scripts/mig_run.sh --wait --venv .venv-eval python official_content/evaluation.py

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[MIG]${NC} $*" >&2; }
log_warn()  { echo -e "${YELLOW}[MIG]${NC} $*" >&2; }
log_error() { echo -e "${RED}[MIG]${NC} $*" >&2; }

WAIT_MODE=0
WAIT_INTERVAL=10
WAIT_TIMEOUT=3600
VENV_PATH=""

# オプション解析
while [[ "$1" == --* ]]; do
    case "$1" in
        --wait)
            WAIT_MODE=1
            shift
            ;;
        --wait-interval=*)
            WAIT_INTERVAL="${1#*=}"
            shift
            ;;
        --wait-timeout=*)
            WAIT_TIMEOUT="${1#*=}"
            shift
            ;;
        --venv=*)
            VENV_PATH="${1#*=}"
            shift
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --list)
            nvidia-smi -L
            echo ""
            echo "--- 使用中プロセス ---"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
                --format=csv,noheader 2>/dev/null || echo "(なし)"
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            exit 1
            ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "使い方: $0 [オプション] <コマンド> [引数...]"
    echo ""
    echo "オプション:"
    echo "  --venv <path>         venv を activate してから実行 (例: .venv-eval)"
    echo "  --wait                空きインスタンスが出るまで待機"
    echo "  --wait-interval=N     待機間隔(秒) [デフォルト: 10]"
    echo "  --wait-timeout=N      最大待機時間(秒) [デフォルト: 3600]"
    echo "  --list                MIG インスタンスの状態一覧を表示"
    exit 1
fi

# 全 MIG インスタンスの UUID と整数インデックスをリスト化 (形式: "0 MIG-uuid")
get_all_mig_devices() {
    local idx=0
    while IFS= read -r line; do
        local uuid
        uuid=$(echo "$line" | grep -oP '(?<=\(UUID: )MIG-[^)]+' || true)
        if [ -n "$uuid" ]; then
            echo "$idx $uuid"
            idx=$((idx + 1))
        fi
    done < <(nvidia-smi -L 2>/dev/null)
}

# 使用中の GPU UUID を取得
get_busy_uuids() {
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' || true
}

# 空き MIG インスタンスのインデックスを1つ選択して出力
find_free_mig() {
    local devices busy_uuids idx uuid

    devices=$(get_all_mig_devices)

    if [ -z "$devices" ]; then
        return 1
    fi

    busy_uuids=$(get_busy_uuids)

    while IFS=" " read -r idx uuid; do
        if [ -z "$uuid" ]; then
            continue
        fi
        if ! echo "$busy_uuids" | grep -qF "$uuid"; then
            echo "$idx"  # 整数インデックスを返す
            return 0
        fi
    done <<< "$devices"

    return 1
}

# 空き MIG インスタンスを探す (--wait 時はポーリング)
if [ "$WAIT_MODE" -eq 1 ]; then
    elapsed=0
    log_info "空き MIG インスタンスを待機中... (最大 ${WAIT_TIMEOUT}s)"
    while true; do
        free_uuid=$(find_free_mig 2>/dev/null) && break
        if [ "$elapsed" -ge "$WAIT_TIMEOUT" ]; then
            log_error "タイムアウト: 空き MIG インスタンスが見つかりませんでした (${WAIT_TIMEOUT}s)"
            exit 1
        fi
        log_warn "空きなし — ${WAIT_INTERVAL}s 後に再試行... (経過: ${elapsed}s)"
        sleep "$WAIT_INTERVAL"
        elapsed=$((elapsed + WAIT_INTERVAL))
    done
else
    free_uuid=$(find_free_mig 2>/dev/null) || {
        log_error "空き MIG インスタンスが見つかりません。"
        echo "" >&2
        echo "現在の状態:" >&2
        nvidia-smi -L >&2
        echo "" >&2
        echo "使用中プロセス:" >&2
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name \
            --format=csv,noheader 2>/dev/null >&2 || echo "(取得失敗)" >&2
        echo "" >&2
        echo "ヒント: --wait オプションで空きが出るまで待機できます" >&2
        exit 1
    }
fi

export CUDA_VISIBLE_DEVICES="$free_uuid"
log_info "使用インスタンス: device $free_uuid"
log_info "実行: $*"
echo "" >&2

# --venv が指定されていれば activate して PATH/環境変数を引き継ぐ
if [ -n "$VENV_PATH" ]; then
    activate="${VENV_PATH}/bin/activate"
    if [ ! -f "$activate" ]; then
        log_error "venv activate スクリプトが見つかりません: $activate"
        exit 1
    fi
    # shellcheck source=/dev/null
    source "$activate"
    log_info "venv: $VENV_PATH"
fi

exec "$@"
