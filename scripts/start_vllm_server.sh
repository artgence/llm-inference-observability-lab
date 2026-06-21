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

append_boolean_flag() {
  local value="$1"
  local enabled_flag="$2"
  local disabled_flag="$3"
  case "$value" in
    "") ;;
    true|TRUE|True|1|yes|YES|Yes) args+=("$enabled_flag") ;;
    false|FALSE|False|0|no|NO|No) args+=("$disabled_flag") ;;
    *)
      echo "Invalid boolean value '$value' for $enabled_flag" >&2
      return 1
      ;;
  esac
}

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

if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
  args+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

if [[ -n "${MAX_NUM_PARTIAL_PREFILLS:-}" ]]; then
  args+=(--max-num-partial-prefills "$MAX_NUM_PARTIAL_PREFILLS")
fi

if [[ -n "${MAX_LONG_PARTIAL_PREFILLS:-}" ]]; then
  args+=(--max-long-partial-prefills "$MAX_LONG_PARTIAL_PREFILLS")
fi

if [[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ]]; then
  args+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
fi

if [[ -n "${KV_CACHE_DTYPE:-}" ]]; then
  args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi

if [[ -n "${CPU_OFFLOAD_GB:-}" ]]; then
  args+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi

append_boolean_flag \
  "${ENABLE_PREFIX_CACHING:-}" \
  --enable-prefix-caching \
  --no-enable-prefix-caching

append_boolean_flag \
  "${ENABLE_CHUNKED_PREFILL:-}" \
  --enable-chunked-prefill \
  --no-enable-chunked-prefill

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
