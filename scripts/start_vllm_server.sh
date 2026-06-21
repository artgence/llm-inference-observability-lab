#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_BIN="${VLLM_BIN:-vllm}"

args=(
  "$VLLM_BIN" serve
  --model "$MODEL_ID"
  --max-model-len "$MAX_MODEL_LEN"
  --port "$PORT"
)

if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  args+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "${HOST:-}" ]]; then
  args+=(--host "$HOST")
fi

if [[ -n "${DTYPE:-}" ]]; then
  args+=(--dtype "$DTYPE")
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  args+=(--api-key "$OPENAI_API_KEY")
fi

if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  args+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

if [[ -n "${GENERATION_CONFIG:-}" ]]; then
  args+=(--generation-config "$GENERATION_CONFIG")
fi

if [[ -n "${REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$REASONING_PARSER")
fi

if [[ "${LANGUAGE_MODEL_ONLY:-false}" == "true" ]]; then
  args+=(--language-model-only)
fi

exec "${args[@]}"
