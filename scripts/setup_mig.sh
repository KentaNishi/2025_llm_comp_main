#!/bin/bash
# MIG (Multi-Instance GPU) セットアップスクリプト
# NVIDIA B200 (183 GiB) 対応
#
# 使い方:
#   ./scripts/setup_mig.sh [profile]
#
# profile オプション:
#   train_eval   : 訓練用 4g.90gb + 評価用 3g.90gb (デフォルト)
#   half_half    : 3g.90gb x 2 (均等分割)
#   quad         : 1g.45gb x 4 (小〜中規模並列)
#   max_split    : 1g.23gb x 7 (最大分割数)
#   single_large : 4g.90gb x 1 (大容量シングル)
#   disable      : MIG を無効化して通常モードに戻す
#   status       : 現在の MIG 状態を表示

set -e

GPU_ID="${GPU_ID:-0}"
PROFILE="${1:-train_eval}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $*"; }

show_status() {
    echo ""
    echo "=========================================="
    echo " GPU MIG 状態"
    echo "=========================================="
    nvidia-smi -i "$GPU_ID"
    echo ""
    echo "--- MIG インスタンス一覧 ---"
    nvidia-smi mig -lgi 2>/dev/null || echo "(MIG インスタンスなし)"
    echo ""
    echo "--- コンピュートインスタンス一覧 ---"
    nvidia-smi mig -lci 2>/dev/null || echo "(コンピュートインスタンスなし)"
    echo ""
    echo "--- CUDA デバイス一覧 ---"
    nvidia-smi -L
}

check_running_processes() {
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null || true)
    if [ -n "$procs" ]; then
        log_warn "GPU を使用中のプロセスが存在します:"
        echo "$procs"
        echo ""
        log_warn "MIG の有効化には GPU が空である必要があります。"
        log_warn "プロセスを終了させてから再実行してください。"
        log_warn "強制的に続行するには: GPU_FORCE=1 $0 $PROFILE"
        if [ "${GPU_FORCE:-0}" != "1" ]; then
            exit 1
        fi
        log_warn "GPU_FORCE=1 が設定されているため続行します..."
    fi
}

enable_mig() {
    log_step "MIG モードを有効化..."
    nvidia-smi -i "$GPU_ID" -mig 1
    log_info "MIG 有効化完了"
}

disable_mig() {
    log_step "既存の MIG インスタンスをすべて削除..."
    nvidia-smi mig -dci 2>/dev/null || true
    nvidia-smi mig -dgi 2>/dev/null || true
    log_step "MIG モードを無効化..."
    nvidia-smi -i "$GPU_ID" -mig 0
    log_info "MIG 無効化完了 — 通常モードに戻りました"
}

create_instances() {
    local profile_id="$1"
    local count="$2"
    local desc="$3"

    log_step "GPU インスタンス作成: プロファイル=$desc x $count"
    for i in $(seq 1 "$count"); do
        nvidia-smi mig -cgi "$profile_id" -C
    done
    log_info "GPU インスタンス作成完了"
}

create_mixed_instances() {
    # 異なるプロファイルを組み合わせる場合
    local profile1_id="$1"
    local desc1="$2"
    local profile2_id="$3"
    local desc2="$4"

    log_step "GPU インスタンス作成: $desc1 + $desc2"
    nvidia-smi mig -cgi "$profile1_id" -C
    nvidia-smi mig -cgi "$profile2_id" -C
    log_info "GPU インスタンス作成完了"
}

# ===== メイン処理 =====

echo ""
echo "=========================================="
echo " MIG セットアップ (GPU $GPU_ID: NVIDIA B200)"
echo " プロファイル: $PROFILE"
echo "=========================================="
echo ""

case "$PROFILE" in
    status)
        show_status
        exit 0
        ;;

    disable)
        check_running_processes
        disable_mig
        show_status
        log_info "完了! GPU は通常モードで使用できます。"
        ;;

    train_eval)
        # 訓練用 4g.90gb (90GB, 72 SM) + 評価用 3g.90gb (90GB, 70 SM)
        # 合計: 180GB をほぼ均等に分割しつつ SM 数で優先度を分ける
        check_running_processes
        enable_mig
        log_step "train_eval 構成: 4g.90gb (訓練) + 3g.90gb (評価)"
        nvidia-smi mig -cgi 5 -C   # 4g.90gb
        nvidia-smi mig -cgi 9 -C   # 3g.90gb
        show_status
        log_info "完了!"
        echo ""
        echo "--- 使用方法 ---"
        echo "  CUDA_VISIBLE_DEVICES=MIG-<UUID-0>  # 訓練用 (4g.90gb)"
        echo "  CUDA_VISIBLE_DEVICES=MIG-<UUID-1>  # 評価用 (3g.90gb)"
        echo "  ※ UUID は上記 'CUDA デバイス一覧' を参照"
        ;;

    half_half)
        # 3g.90gb x 2 (均等2分割)
        check_running_processes
        enable_mig
        log_step "half_half 構成: 3g.90gb x 2"
        nvidia-smi mig -cgi 9 -C
        nvidia-smi mig -cgi 9 -C
        show_status
        log_info "完了! 90GB x 2 インスタンスが作成されました。"
        ;;

    quad)
        # 1g.45gb x 4 (45GB x 4 = 180GB)
        check_running_processes
        enable_mig
        log_step "quad 構成: 1g.45gb x 4"
        for i in 1 2 3 4; do
            nvidia-smi mig -cgi 15 -C
        done
        show_status
        log_info "完了! 45GB x 4 インスタンスが作成されました。"
        echo ""
        echo "--- 並列評価例 ---"
        echo "  nvidia-smi -L で各インスタンスの UUID を確認してください"
        ;;

    max_split)
        # 1g.23gb x 7 (約23GB x 7 = 最大7プロセス並列)
        check_running_processes
        enable_mig
        log_step "max_split 構成: 1g.23gb x 7"
        for i in 1 2 3 4 5 6 7; do
            nvidia-smi mig -cgi 19 -C
        done
        show_status
        log_info "完了! 23GB x 7 インスタンスが作成されました。"
        echo ""
        echo "--- 並列評価例 ---"
        echo "  for i in {0..6}; do"
        echo "    CUDA_VISIBLE_DEVICES=MIG-<UUID-\$i> python eval.py & "
        echo "  done; wait"
        ;;

    single_large)
        # 4g.90gb x 1 (大容量シングル、残り 3g.90gb 分は未使用)
        check_running_processes
        enable_mig
        log_step "single_large 構成: 4g.90gb x 1"
        nvidia-smi mig -cgi 5 -C
        show_status
        log_info "完了! 90GB シングルインスタンスが作成されました。"
        ;;

    *)
        log_error "不明なプロファイル: $PROFILE"
        echo ""
        echo "使用可能なプロファイル:"
        echo "  train_eval   : 4g.90gb (訓練) + 3g.90gb (評価) [デフォルト]"
        echo "  half_half    : 3g.90gb x 2"
        echo "  quad         : 1g.45gb x 4"
        echo "  max_split    : 1g.23gb x 7"
        echo "  single_large : 4g.90gb x 1"
        echo "  disable      : MIG 無効化"
        echo "  status       : 状態確認"
        exit 1
        ;;
esac
