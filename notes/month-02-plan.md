# Month 2 Plan: Operating-Point Robustness

Goal: use the Month 1 operating point of 18 RPS to determine how prompt length, output length, and burst shape change prefill, decode, queueing, GPU-memory, and cost behavior.

## Implementation Status

- [x] Use the Month 1 best-performing steady rate of 18 RPS for Month 2.
- [x] Prompt-token target sweep at 64, 128, 256, 512, and 1024 at 18 RPS.
- [x] Output-token target sweep at 64, 128, 256, 512, and 1024 at 18 RPS.
- [x] Steady-versus-bursty open-loop comparison at 18 average RPS for p512/o128, p512/o256, and p512/o512.
- [x] TTFT, TPOT, end-to-end latency, throughput, error, timeout, OOM, and GPU metrics.
- [x] Configurable GPU cost and cost-per-million-token estimates.
- [x] Automated operating-point degradation and bottleneck analysis.
- [ ] Run every workload on the L40S.
- [ ] Review actual prompt/output token counts and reject invalid runs.
- [ ] Complete Report 02 with measured results and conclusions.

## Server Prerequisite

The default `MAX_MODEL_LEN=8192` has sufficient context capacity for the largest 1024-token prompt and 1024-token output stages, including chat-template overhead.

## Run Sequence

Set the effective hourly GPU price if cost-per-token estimates are required:

```bash
export GPU_HOURLY_COST_USD=1.00  # Replace with the effective L40S hourly cost.
```

Run each controlled sweep separately so it gets an independent run directory:

```bash
python3 scripts/benchmark_vllm.py --workload workloads/month2_prompt_length_sweep.json
python3 scripts/benchmark_vllm.py --workload workloads/month2_output_length_sweep.json
python3 scripts/benchmark_vllm.py --workload workloads/month2_burst_sweep.json
```

Generate the combined analysis by passing the three resulting directories:

```bash
python3 scripts/analyze_saturation.py \
  benchmarks/PROMPT_RUN_ID \
  benchmarks/OUTPUT_RUN_ID \
  benchmarks/BURST_RUN_ID \
  --out reports/report-02-results.md
```

## Acceptance Checks

- `latency_s` and TTFT start immediately before the HTTP send attempt; `scheduled_latency_s` separately includes time since the planned open-loop arrival.
- `prompt_tokens_avg` is reasonably close to `prompt_tokens_target`; use the server-reported average in conclusions.
- `output_tokens_avg` is reasonably close to `output_tokens_target`; otherwise the output-length stage is not valid.
- `scheduler_delay_p95_s` remains below 0.1 seconds for open-loop stages. Otherwise the load generator, not vLLM, is limiting arrivals.
- Every stage has enough successful requests for percentile interpretation.
- Cost conclusions state the supplied GPU hourly price and exclude non-GPU infrastructure costs.
- Any claim that 18 RPS remains acceptable under a changed workload shape is supported by p99 latency, throughput, errors, and GPU evidence rather than a single metric.
