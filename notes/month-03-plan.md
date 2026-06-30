# Month 3 Plan: Observability and Failure Diagnosis

Goal: correlate client behavior, vLLM scheduler/cache state, and GPU telemetry well enough to diagnose production-style failures.

## Implementation Status

- [x] Continuous run-correlated vLLM `/metrics` capture.
- [x] Request-level logs, run IDs, workload metadata, error classification, timeout tracking, and GPU sampling.
- [x] Optional live benchmark Prometheus exporter for failures, timeouts, OOMs, rejections, and client in-flight requests.
- [x] Prometheus, Grafana, and NVIDIA DCGM Exporter Compose stack.
- [x] Provisioned dashboard covering the Month 3 metric checklist.
- [x] Alert rules for metrics loss, queueing, KV pressure, TTFT, failures, timeouts, and OOMs.
- [x] Traffic-burst, long-prompt-storm, and memory-pressure incident workloads.
- [x] Automated incident-note generator and Report 03 template.
- [ ] Execute the three incidents on the L40S and preserve screenshots or exported panels.
- [ ] Review generated incident notes and complete Report 03.

## Start Observability

```bash
docker compose -f observability/docker-compose.yml up -d
```

## Incident 1: Traffic Burst

Compare steady traffic with burst sizes 16, 32, and 64 at average rates of 18, 24, 30, and 36 RPS. Each case runs for 20 seconds at p512/o128.

Use the normal 8K server configuration:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month3_traffic_burst_incident.json \
  --metrics-export-port 9001
```

## Incident 2: Long Prompt Storm

Use an ascending prompt-target sweep of 2,048, 4,096, 8,192, 12,288, and 15,360 tokens with a fixed 128-token output target. Every stage stays below the 16,384-token server limit; the largest leaves 896 tokens for output and chat-template/token-estimation overhead. Configure the Runpod startup command with `--max-model-len 16384` before creating the pod. Do not start a second vLLM process while PID 1 owns port 8000.

Then run:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month3_long_prompt_storm_incident.json \
  --expect-server-config max_model_len=16384 \
  --metrics-export-port 9001
```

The workload-level `max_model_len` guard rejects any stage whose prompt target plus output cap reaches or exceeds 16,384 tokens.

## Incident 3: Memory Pressure

Recreate the pod with `--max-model-len 16384` in its startup command, then deliberately increase concurrent KV demand:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month3_memory_pressure_incident.json \
  --expect-server-config max_model_len=16384 \
  --metrics-export-port 9001
```

This experiment is intentionally capable of causing OOM or severe queueing. Run it only on the isolated lab server.

## Generate Incident Notes

```bash
python3 scripts/diagnose_incident.py benchmarks/TRAFFIC_BURST_RUN_ID --incident-type traffic_burst --out incidents/incident-01-traffic-burst.md
python3 scripts/diagnose_incident.py benchmarks/LONG_PROMPT_RUN_ID --incident-type long_prompt_storm --out incidents/incident-02-long-prompt-storm.md
python3 scripts/diagnose_incident.py benchmarks/MEMORY_PRESSURE_RUN_ID --incident-type memory_pressure --out incidents/incident-03-memory-pressure.md
```

## Acceptance Checks

- `vllm_metrics_sample_errors` is zero or the missing telemetry is explained.
- Dashboard time range covers the full incident and drain period.
- Queue/KV conclusions cite `vllm_requests_waiting_max` and `vllm_kv_cache_usage_pct_max` when available.
- Client failure panels use the same run ID as the benchmark artifacts.
- Every incident note distinguishes observed evidence from inference.
