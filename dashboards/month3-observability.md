# Month 3 Observability Dashboard

The provisioned Grafana dashboard is stored at `observability/grafana/dashboards/vllm-observability.json`.

It covers:

- completed request rate
- active and queued vLLM requests
- TTFT p50/p95/p99
- TPOT p50/p95/p99
- end-to-end latency p95/p99
- prompt and generation token throughput
- KV-cache usage
- GPU framebuffer memory and utilization
- benchmark failure rate and client in-flight requests
- timeout, OOM, and rejected-request counts

The vLLM panels use current V1 metric names. If a different vLLM release exposes legacy names, inspect `/metrics` and update the PromQL rather than silently treating an empty panel as zero.

The timeout/OOM/rejection panels require running the benchmark with `--metrics-export-port 9001`. Prometheus retains those series after the benchmark exits, but the exporter target will correctly show as down between runs.
