# Month 1 Metrics Notes

Month 1 keeps dashboard work lightweight. Use the benchmark result files as the source of truth, then compare them with vLLM's `/metrics` endpoint.

Useful vLLM metrics to inspect during baseline runs:

- `vllm:e2e_request_latency_seconds`
- `vllm:time_to_first_token_seconds`
- `vllm:request_time_per_output_token_seconds`
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:kv_cache_usage_perc`
- `vllm:request_success`
- `vllm:prompt_tokens`
- `vllm:generation_tokens`
- `vllm:prefix_cache_hits`
- `vllm:prefix_cache_queries`

For the first report, include raw benchmark metrics first. Prometheus/Grafana becomes the main workstream in Month 3.
