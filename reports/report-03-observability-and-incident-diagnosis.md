# Report 03: Observability and Incident Diagnosis for vLLM Serving

## Summary

Status: awaiting Month 3 GPU-backed incident runs.

This report correlates request-level benchmark data, vLLM Prometheus metrics, and NVIDIA GPU telemetry for three controlled incidents.

## Environment

| Field | Value |
| --- | --- |
| Date | TBD |
| GPU | NVIDIA L40S |
| Model | neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 |
| vLLM version | TBD |
| Prometheus version | TBD |
| Grafana version | TBD |
| Dashboard UID | vllm-observability-lab |

## Observability Coverage

| Signal | Source | Status |
| --- | --- | --- |
| Request rate, running/waiting, TTFT, TPOT, E2E latency, token throughput, KV cache | vLLM `/metrics` | TBD |
| Failure rate, timeouts, OOMs, rejections, client in-flight | Benchmark exporter | TBD |
| GPU memory and utilization | NVIDIA DCGM Exporter and run CSV | TBD |
| Request details and classifications | `requests.jsonl` | TBD |
| Run/workload correlation | metadata and `vllm_metrics.jsonl` | TBD |

## Incident 1: Traffic Burst

Link: `incidents/incident-01-traffic-burst.md`

- Test matrix: steady and burst sizes 16, 32, and 64 at 18, 24, 30, and 36 average RPS; p512/o128
- Symptom and impact: TBD
- Queue/TTFT evidence: TBD
- Mitigation and alert: TBD

## Incident 2: Long Prompt Storm

Link: `incidents/incident-02-long-prompt-storm.md`

- Symptom and impact: TBD
- Prefill, queue, and KV evidence: TBD
- Mitigation and alert: TBD

## Incident 3: Memory Pressure

Link: `incidents/incident-03-memory-pressure.md`

- Symptom and impact: TBD
- KV-cache, GPU-memory, OOM, and rejection evidence: TBD
- Mitigation and alert: TBD

## Cross-Incident Findings

- Earliest reliable saturation signal: TBD
- Metric that best distinguishes prefill from decode pressure: TBD
- Safe operating control: TBD
- Dashboard or alert gap discovered: TBD

## Conclusion

- What failed first: TBD
- What would have reduced impact: TBD
- What should be monitored continuously: TBD
