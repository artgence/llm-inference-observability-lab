# Replica Routing Experiment

This module routes OpenAI-compatible streaming requests across independent vLLM
replicas. It models inference data parallelism: one request is assigned to one
replica, and every replica holds a full model copy.

Do not start extra vLLM processes inside a Runpod image whose PID 1 already owns its
GPU and port. Use two separately configured pods/endpoints, or use a clean base image
where each replica has an explicitly assigned GPU.

## Start the Router

```bash
python3 routing/router.py \
  --worker replica_a=http://REPLICA_A:8000 \
  --worker replica_b=http://REPLICA_B:8000 \
  --policy round_robin \
  --port 9000
```

Supported policies:

- `round_robin`: deterministic even distribution.
- `least_inflight`: selects the worker with the fewest active requests.
- `latency_aware`: minimizes EWMA response latency multiplied by current load.

The router preserves streaming responses and exposes `/health` and `/metrics`.
Retries are disabled by default because retrying a generation request can duplicate
work. For the controlled failure experiment only, `--max-retries 1` permits one retry
on another replica before any response has been sent to the client.

## Benchmark Through the Router

```bash
VLLM_BASE_URL=http://127.0.0.1:9000 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_replica_routing.json \
  --server-config-label replicas_round_robin \
  --server-launch-command \
    'python3 routing/router.py --worker replica_a=http://REPLICA_A:8000 --worker replica_b=http://REPLICA_B:8000 --policy round_robin --port 9000' \
  --gpu-hourly-cost-usd TOTAL_COST_OF_BOTH_REPLICAS
```

Repeat with `least_inflight` and `latency_aware`. The benchmark drain guard works
through the router because router `/metrics` exports aggregate
`vllm:num_requests_running` and `vllm:num_requests_waiting` gauges.

If the replicas are remote pods, collect GPU memory/utilization on each replica.
The benchmark's local `gpu_metrics.csv` describes only the router/benchmark host and
must not be presented as remote replica GPU balance.

## Slow or Failing Replica

Place the fault proxy in front of replica B:

```bash
python3 routing/slow_worker_proxy.py \
  --upstream http://REPLICA_B:8000 \
  --port 8102 \
  --delay-ms 500 \
  --fail-every 4

python3 routing/router.py \
  --worker replica_a=http://REPLICA_A:8000 \
  --worker replica_b=http://127.0.0.1:8102 \
  --policy latency_aware \
  --max-retries 1 \
  --failure-threshold 2 \
  --circuit-open-seconds 15 \
  --port 9000
```

Compare successful RPS, p95/p99, timeout/error rate, retry count, per-worker request
balance, and circuit-open behavior. This is a bounded infrastructure experiment, not
an advanced request-routing or distributed KV-cache implementation.
