# Month 4 Plan: Serving Knobs and Capacity Boundaries

Goal: measure how vLLM scheduling, cache, memory, and admission knobs change TTFT, TPOT, throughput, failures, and cost.

## Implementation Status

- [x] Add explicit launcher controls for prefix caching, chunked prefill, batching limits, KV-cache dtype, GPU-memory utilization, CPU offload, and quantization.
- [x] Add repeated-versus-unique prefix generation with equal prompt lengths.
- [x] Add continuous-batching, context-capacity, admission-control, and quantized-70B workloads.
- [x] Add workload-level mixed prompt lengths and reject/truncate admission policies.
- [x] Add server-configuration labels and a Month 4 comparison report generator.
- [ ] Define a small task-quality evaluation set before claiming an 8B/70B quality tradeoff.
- [ ] Execute each server variant on the GPU host.
- [ ] Complete Report 04 with measured boundaries and selected production defaults.

Do not launch a second vLLM process while the existing PID 1 server owns port 8000. Restart or recreate the serving container between server-side variants. Use the same model, workload, GPU cost, and benchmark options for both sides of each comparison.

## Experiment 1: Prefix Caching

The workload compares a request-unique 1,536-token prefix with a shared 1,536-token prefix inside equal 2,048-token prompts. Run it once with automatic prefix caching explicitly disabled and once enabled.

```bash
ENABLE_PREFIX_CACHING=false scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_prefix_cache.json \
  --server-config-label prefix_cache_off

ENABLE_PREFIX_CACHING=true scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_prefix_cache.json \
  --server-config-label prefix_cache_on
```

Compare TTFT and `vllm_prefix_cache_hit_rate` first. Automatic prefix caching reuses prefill KV blocks; it should not be presented as a decode/TPOT optimization.

## Experiment 2: Chunked Prefill and Long Context

Run the same 2K, 8K, and 15,360-token prompt sweep with chunked prefill disabled and enabled. Keep `MAX_NUM_BATCHED_TOKENS` fixed so the comparison is attributable.

```bash
ENABLE_CHUNKED_PREFILL=false MAX_NUM_BATCHED_TOKENS=8192 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_context_capacity.json \
  --server-config-label chunked_prefill_off

ENABLE_CHUNKED_PREFILL=true \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_PARTIAL_PREFILLS=2 \
MAX_LONG_PARTIAL_PREFILLS=1 \
LONG_PREFILL_TOKEN_THRESHOLD=4096 \
scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_context_capacity.json \
  --server-config-label chunked_prefill_on
```

## Experiment 3: Continuous Batching and Capacity

Run the same concurrency sweep with explicit sequence limits. Compare where throughput flattens and TTFT, queue depth, or KV-cache pressure accelerates.

```bash
MAX_NUM_SEQS=32 GPU_MEMORY_UTILIZATION=0.90 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_batching_capacity.json \
  --server-config-label max_seqs_32

MAX_NUM_SEQS=64 GPU_MEMORY_UTILIZATION=0.90 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_batching_capacity.json \
  --server-config-label max_seqs_64
```

## Experiment 4: Admission Control

This client-side simulation repeats a mixed prompt pattern of 512, 512, 2,048, and 15,360 tokens. It compares no policy, rejecting prompts over 8,192 tokens, and truncating them to 8,192 tokens.

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_admission_control.json \
  --server-config-label admission_control
```

`admission_rejected_count` is an intentional policy result. Do not combine it with server OOM, timeout, or 5xx failures when judging reliability.

## Experiment 5: Quantized 70B

The FP8 70B checkpoint is approximately half the BF16 weight footprint but still does not fit in one 46 GB L40S. Use at least two L40S GPUs and tensor parallelism; skip this experiment when only one GPU is available.

```bash
MODEL_ID=neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=16384 \
scripts/start_vllm_server.sh

MODEL_ID=neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_quantized_70b.json \
  --server-config-label llama31_70b_fp8_tp2
```

## Build Report 04

Supply baseline variants before candidate variants so percentage comparisons use the intended reference.

```bash
python3 scripts/analyze_month4.py \
  benchmarks/PREFIX_OFF_RUN \
  benchmarks/PREFIX_ON_RUN \
  benchmarks/CHUNKED_OFF_RUN \
  benchmarks/CHUNKED_ON_RUN \
  benchmarks/MAX_SEQS_32_RUN \
  benchmarks/MAX_SEQS_64_RUN \
  benchmarks/ADMISSION_RUN \
  --out reports/report-04-results.md
```

## Acceptance Checks

- All server variants record a distinct `server_config_label` in `metadata.json`.
- Prefix prompts have the same total target and differ only in prefix reuse.
- Prefix-cache claims include non-zero cache query/hit evidence from vLLM metrics.
- Scheduler-delay p95 remains low enough that the client is not the bottleneck.
- Context targets plus output caps remain below `MAX_MODEL_LEN=16384`.
- Any claimed improvement includes TTFT/TPOT, throughput, errors, waiting requests, KV usage, GPU memory, and cost evidence.
- The chosen serving defaults state the tested GPU/model boundary and do not generalize a two-GPU 70B result to one L40S.

## References

- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM quantization support](https://docs.vllm.ai/en/stable/features/quantization/index.html)
