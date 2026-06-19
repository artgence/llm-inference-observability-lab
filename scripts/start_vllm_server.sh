#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
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
  --generation-config vllm
)

if [[ -n "$REASONING_PARSER" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "$LANGUAGE_MODEL_ONLY" == "true" ]]; then
  args+=(--language-model-only)
fi

exec "${args[@]}"
