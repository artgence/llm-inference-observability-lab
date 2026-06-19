# Report 01: Baseline vLLM Serving on L40S

## Summary

Status: awaiting first GPU-backed run.

This report captures the first reproducible baseline for vLLM serving using meta-llama/Llama-3.1-8B-Instruct, OpenAI-compatible requests, request-level latency metrics, token throughput, error tracking, and GPU telemetry.

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
| Model | meta-llama/Llama-3.1-8B-Instruct |

## Server Command

```bash
MODEL_ID=meta-llama/Llama-3.1-8B-Instruct scripts/start_vllm_server.sh
```

Record the exact expanded command and any deviations here. For full native Llama 3.1 context, note `MAX_MODEL_LEN` and `TENSOR_PARALLEL_SIZE`.

## Workload

Workload file: `workloads/month1_baseline.json`

The baseline workload uses the model's standard instruct chat template with deterministic sampling so TTFT, TPOT, and output length are easier to compare across runs.

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
- Decide whether to keep the small 8K context baseline or run a larger Llama 3.1 context test if GPU memory allows.
