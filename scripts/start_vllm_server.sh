#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

exec vllm serve "$MODEL_ID" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --api-key "$API_KEY" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --generation-config vllm
