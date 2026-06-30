# Report 01: Baseline vLLM Serving on L40S

## Summary

Status: awaiting first GPU-backed run.

This report captures the first reproducible baseline for vLLM serving using neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8, OpenAI-compatible requests, request-level latency metrics, token throughput, error tracking, and GPU telemetry.

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
| Model | neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 |

## Server Command

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --max-model-len 16384 \
  --port 8000
```

Use this as the Runpod container startup command. Copy the command discovered in
benchmark `metadata.json` here, together with the captured
`vllm:cache_config_info` labels and any deviations.

## Workload

Workload files: `workloads/month1_baseline.json` and `workloads/month1_open_loop.json`

The baseline workload uses the model's standard instruct chat template with deterministic sampling so TTFT, TPOT, and output length are easier to compare across runs.

| Workload | Requests | Concurrency | Prompt Words | Max Output Tokens |
| --- | ---: | ---: | ---: | ---: |
| smoke_c1_prompt256_out128 | 5 | 1 | 256 | 128 |
| baseline_c4_prompt512_out128 | 20 | 4 | 512 | 128 |
| baseline_c8_prompt512_out128 | 40 | 8 | 512 | 128 |
| baseline_c16_prompt512_out128 | 80 | 16 | 512 | 128 |
| baseline_c32_prompt512_out128 | 160 | 32 | 512 | 128 |
| baseline_c48_prompt512_out128 | 240 | 48 | 512 | 128 |
| baseline_c64_prompt512_out128 | 320 | 64 | 512 | 128 |
| baseline_c96_prompt512_out128 | 480 | 96 | 512 | 128 |

| Open-Loop Workload | Requests | Target RPS | Duration | Max In Flight |
| --- | ---: | ---: | ---: | ---: |
| open_loop_rps5_prompt512_out128 | 100 | 5 | 20s | 256 |
| open_loop_rps10_prompt512_out128 | 200 | 10 | 20s | 256 |
| open_loop_rps12_prompt512_out128 | 240 | 12 | 20s | 256 |
| open_loop_rps14_prompt512_out128 | 280 | 14 | 20s | 256 |
| open_loop_rps16_prompt512_out128 | 320 | 16 | 20s | 256 |
| open_loop_rps18_0_prompt512_out128 | 360 | 18.0 | 20s | 256 |
| open_loop_rps18_5_prompt512_out128 | 370 | 18.5 | 20s | 256 |
| open_loop_rps19_0_prompt512_out128 | 380 | 19.0 | 20s | 256 |
| open_loop_rps19_5_prompt512_out128 | 390 | 19.5 | 20s | 256 |
| open_loop_rps20_0_prompt512_out128 | 400 | 20.0 | 20s | 256 |
| open_loop_rps30_prompt512_out128 | 600 | 30 | 20s | 256 |

Report target and achieved request start rates, completion throughput, scheduler delay, latency, and errors separately for the open-loop stages.

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
