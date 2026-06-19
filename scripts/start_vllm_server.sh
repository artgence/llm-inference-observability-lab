#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

args=(
  vllm serve "$MODEL_ID"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype "$DTYPE"
  --api-key "$API_KEY"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --reasoning-parser "$REASONING_PARSER"
  --generation-config vllm
)

if [[ "$LANGUAGE_MODEL_ONLY" == "true" ]]; then
  args+=(--language-model-only)
fi

exec "${args[@]}"
