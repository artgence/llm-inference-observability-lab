# Month 1 Plan

Goal: build a clean, reproducible baseline loop before adding complex serving knobs.

## Checklist

- [x] Create project structure.
- [x] Preserve the six-month roadmap in `memory.md`.
- [x] Add vLLM server start command.
- [x] Add OpenAI-compatible smoke test.
- [x] Add benchmark script with streaming TTFT capture.
- [x] Add GPU metrics sampler.
- [x] Add raw `/metrics` scraper.
- [x] Add simple result table generator.
- [x] Add baseline workload config.
- [x] Add Report 01 template.
- [ ] Run on an L40S GPU host.
- [ ] Fill Report 01 with real numbers.

## Baseline Run Flow

Start the server:

```bash
MODEL_ID=Qwen/Qwen3.6-35B-A3B scripts/start_vllm_server.sh
```

Smoke test:

```bash
python3 scripts/openai_smoke_test.py
```

Dry-run the workload:

```bash
python3 scripts/benchmark_vllm.py --workload workloads/month1_baseline.json --dry-run
```

Run the benchmark:

```bash
python3 scripts/benchmark_vllm.py --workload workloads/month1_baseline.json
```

Scrape vLLM metrics before or after a benchmark:

```bash
python3 scripts/scrape_vllm_metrics.py --output benchmarks/metrics.prom
```

## Qwen3.6 Serving Notes

For the first baseline, keep `MAX_MODEL_LEN=8192` unless the GPU host has enough memory for a larger KV cache. For a full native-context Qwen3.6 run, set `MAX_MODEL_LEN=262144` and `TENSOR_PARALLEL_SIZE=8` on suitable multi-GPU hardware. The Month 1 workload disables thinking with `chat_template_kwargs.enable_thinking=false` to keep output length and latency easier to compare.

## First Analysis Questions

- Is TTFT stable at concurrency 1?
- Does p95/p99 widen at concurrency 4?
- Is token throughput improving with concurrency?
- Is GPU memory stable, near full, or climbing toward a failure boundary?
- Are timeouts, rejected requests, or OOMs present?
