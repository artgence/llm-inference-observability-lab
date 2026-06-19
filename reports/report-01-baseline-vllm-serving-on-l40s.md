# Report 01: Baseline vLLM Serving on L40S

## Summary

Status: awaiting first GPU-backed run.

This report captures the first reproducible baseline for vLLM serving using Qwen/Qwen3.6-35B-A3B, OpenAI-compatible requests, request-level latency metrics, token throughput, error tracking, and GPU telemetry.

## Environment

| Field | Value |
| --- | --- |
| Date | TBD |
| GPU | L40S |
| GPU count | TBD |
| CPU / RAM | TBD |
| OS / image | TBD |
| Python | TBD |
| vLLM version | TBD |
| CUDA / driver | TBD |
| Model | Qwen/Qwen3.6-35B-A3B |

## Server Command

```bash
MODEL_ID=Qwen/Qwen3.6-35B-A3B scripts/start_vllm_server.sh
```

Record the exact expanded command and any deviations here. For full native Qwen3.6 context, note `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`, and whether `LANGUAGE_MODEL_ONLY` was enabled.

## Workload

Workload file: `workloads/month1_baseline.json`

The baseline workload disables Qwen3.6 thinking mode with `chat_template_kwargs.enable_thinking=false` so TTFT, TPOT, and output length are easier to compare across runs.

| Workload | Requests | Concurrency | Prompt Words | Max Output Tokens |
| --- | ---: | ---: | ---: | ---: |
| smoke_c1_prompt256_out128 | 5 | 1 | 256 | 128 |
| baseline_c4_prompt512_out128 | 20 | 4 | 512 | 128 |

## Results

Paste `benchmarks/<run_id>/summary.md` below after the first run.

```text
TBD
```

## Bottlenecks Observed

- TBD

## Errors / Timeouts / OOMs

- TBD

## What I Learned

- TBD

## Next Run

- Increase request count enough for a more meaningful p95/p99.
- Add a longer prompt baseline before Month 2 saturation sweeps.
- Decide whether to keep the small 8K context baseline or run a larger Qwen3.6 context test on multi-GPU hardware.
