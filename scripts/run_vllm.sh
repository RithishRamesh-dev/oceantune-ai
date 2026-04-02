#!/usr/bin/env bash
# =============================================================================
# scripts/run_vllm.sh
# OceanTune AI — vLLM Server Launcher
#
# Called by VLLMServer._build_command() or directly for manual debugging:
#
#   ./scripts/run_vllm.sh \
#       --model mistralai/Mistral-7B-Instruct-v0.2 \
#       --port 8000 \
#       --tensor-parallel-size 2 \
#       --gpu-memory-utilization 0.90
#
# Handles: CUDA/ROCm env, ulimits, log routing, PID file, SIGTERM forwarding.
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${OCEANTUNE_LOG_DIR:-$(dirname "$0")/../storage/logs}"
PID_DIR="${OCEANTUNE_PID_DIR:-/tmp/oceantune}"

mkdir -p "$LOG_DIR" "$PID_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/vllm_${PORT}_${TIMESTAMP}.log"
PID_FILE="$PID_DIR/vllm_${PORT}.pid"

# ── System limits ─────────────────────────────────────────────────────────────
ulimit -n 65536 2>/dev/null || true
ulimit -v unlimited 2>/dev/null || true

# ── CUDA / NCCL environment ───────────────────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# ── Signal forwarding ─────────────────────────────────────────────────────────
_VLLM_PID=""
_cleanup() {
    echo "[run_vllm.sh] Received signal — stopping vLLM (PID $_VLLM_PID)" >&2
    if [ -n "$_VLLM_PID" ] && kill -0 "$_VLLM_PID" 2>/dev/null; then
        kill -TERM "$_VLLM_PID" 2>/dev/null || true
        wait "$_VLLM_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    exit 0
}
trap '_cleanup' SIGTERM SIGINT SIGQUIT

# ── Launch vLLM ───────────────────────────────────────────────────────────────
echo "[run_vllm.sh] Launching vLLM on ${HOST}:${PORT}" >&2
echo "[run_vllm.sh] Log file: ${LOG_FILE}" >&2
echo "[run_vllm.sh] Args: $*" >&2

python3 -m vllm.entrypoints.openai.api_server \
    --host "$HOST" \
    --port "$PORT" \
    "$@" \
    2>&1 | tee "$LOG_FILE" &

_VLLM_PID=$!
echo "$_VLLM_PID" > "$PID_FILE"
echo "[run_vllm.sh] vLLM PID: $_VLLM_PID -> $PID_FILE" >&2

wait "$_VLLM_PID"
VLLM_EXIT=$?

rm -f "$PID_FILE"
echo "[run_vllm.sh] vLLM exited with code $VLLM_EXIT" >&2
exit "$VLLM_EXIT"
