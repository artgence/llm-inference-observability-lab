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
MODEL_ID=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 scripts/start_vllm_server.sh
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

Run the open-loop arrival-rate sweep:

```bash
python3 scripts/benchmark_vllm.py --workload workloads/month1_open_loop.json
```

The open-loop workload schedules 5, 10, 12, 14, 16, 18.0, 18.5, 19.0, 19.5, 20.0, and 30 RPS independently of completions, with half-RPS steps near the expected 18–20 RPS saturation boundary. Compare the target rate with `achieved_request_start_rate_rps`, and reject a run as load-generator-limited if `scheduler_delay_p95_s` is material.

Scrape vLLM metrics before or after a benchmark:

```bash
python3 scripts/scrape_vllm_metrics.py --output benchmarks/metrics.prom
```

## Llama 3.1 Serving Notes

The project now defaults to `MAX_MODEL_LEN=16384`; lower it explicitly if the baseline needs a smaller KV cache. Llama 3.1 8B supports a native context length of 131,072 tokens; increase beyond 16,384 only after validating memory headroom on the target GPU. The model runs with `TENSOR_PARALLEL_SIZE=1` on a single L40S. Access to the gated model repository requires accepting Meta's license and authenticating with Hugging Face.

## First Analysis Questions

- Is TTFT stable at concurrency 1?
- Does p95/p99 widen at concurrency 4?
- Is token throughput improving with concurrency?
- Is GPU memory stable, near full, or climbing toward a failure boundary?
- Are timeouts, rejected requests, or OOMs present?
