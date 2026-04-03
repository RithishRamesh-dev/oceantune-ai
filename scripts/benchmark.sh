#!/usr/bin/env bash
# =============================================================================
# scripts/benchmark.sh
# OceanTune AI — vLLM Benchmark Runner
#
# Wraps `vllm bench serve` for manual benchmark runs and debugging.
# The Python BenchmarkEngine calls the Python entry point directly;
# this script is provided for manual testing and CI debugging.
#
# Usage:
#   ./scripts/benchmark.sh \
#       --base-url http://localhost:8000 \
#       --model deepseek-ai/DeepSeek-V3.2 \
#       --input-len 1024 \
#       --output-len 1024 \
#       --concurrency 16 \
#       --num-prompts 200
#
# Full concurrency ramp (calls this script 7 times):
#   for c in 1 2 4 8 16 32 64; do
#       ./scripts/benchmark.sh --concurrency $c [other args]
#   done
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V3.2}"
INPUT_LEN="${INPUT_LEN:-1024}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
CONCURRENCY="${CONCURRENCY:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
LOG_DIR="${OCEANTUNE_LOG_DIR:-$(dirname "$0")/../storage/logs}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/bench_c${CONCURRENCY}_${TIMESTAMP}.log"

# ── Parse CLI args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)   BASE_URL="$2";    shift 2 ;;
        --model)      MODEL="$2";       shift 2 ;;
        --input-len)  INPUT_LEN="$2";   shift 2 ;;
        --output-len) OUTPUT_LEN="$2";  shift 2 ;;
        --concurrency|--max-concurrency) CONCURRENCY="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --log-dir)    LOG_DIR="$2";     shift 2 ;;
        *) echo "Unknown arg: $1" >&2;  exit 1 ;;
    esac
done

echo "[benchmark.sh] Starting benchmark" >&2
echo "  Base URL     : $BASE_URL" >&2
echo "  Model        : $MODEL" >&2
echo "  Input len    : $INPUT_LEN tokens" >&2
echo "  Output len   : $OUTPUT_LEN tokens" >&2
echo "  Concurrency  : $CONCURRENCY" >&2
echo "  Num prompts  : $NUM_PROMPTS" >&2
echo "  Log file     : $LOG_FILE" >&2

# ── Health check before benchmarking ─────────────────────────────────────────
HEALTH_URL="${BASE_URL%/}/health"
echo "[benchmark.sh] Checking server health at $HEALTH_URL ..." >&2

for attempt in 1 2 3; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo "[benchmark.sh] Server is healthy." >&2
        break
    fi
    if [ "$attempt" -eq 3 ]; then
        echo "[benchmark.sh] ERROR: Server not healthy after 3 attempts." >&2
        exit 1
    fi
    echo "[benchmark.sh] Server not ready, retrying in 5s... (attempt $attempt/3)" >&2
    sleep 5
done

# ── Run benchmark ─────────────────────────────────────────────────────────────
python -m vllm.entrypoints.openai.run_bench \
    --backend vllm \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --request-rate inf \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,95,99 \
    --ignore-eos \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[benchmark.sh] Benchmark complete (exit code: $EXIT_CODE). Log: $LOG_FILE" >&2
exit "$EXIT_CODE"