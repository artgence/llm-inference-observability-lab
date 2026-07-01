# Month 5 Plan: Parallelism, Decoding, and Routing Trade-offs for LLM Serving

Goal: answer one production inference question:

> For a fixed model and workload, should the service use one larger GPU, tensor
> parallelism across smaller GPUs, or independent vLLM replicas behind a router?

This is not a distributed-training course or an acronym survey. Every experiment
must explain a model-fit, latency, throughput, failure-isolation, topology, or cost
trade-off.

## Priority and Required Depth

| Topic | Priority | Depth | Required outcome |
| --- | --- | --- | --- |
| Tensor parallelism (TP) | Required | Medium-deep | Explain model fit, KV capacity, collectives, topology, and measured latency/cost overhead |
| Data-parallel replicas | Required | Medium-deep | Run independent replicas, route one request to one replica, and measure throughput/failure isolation |
| NCCL | Required | Medium | Explain all-reduce/all-gather/reduce-scatter/broadcast, topology sensitivity, and common failure classes |
| Pipeline parallelism (PP) | Optional | Shallow-medium | Know when fit/topology may justify it; one optional run only |
| Expert parallelism (EP) | Conceptual | Shallow | Explain MoE expert placement and imbalance without implementing it |
| Speculative decoding | Required bounded experiment | Medium | Measure acceptance, TPOT, throughput, memory, and saturation behavior |

Explicitly out of scope:

- training-style DDP internals
- multi-node TP unless a role specifically requires it
- custom schedulers or CUDA/NCCL kernels
- advanced KV-aware routing, distributed KV migration, or routing research
- prefill/decode disaggregation as a Month 5 requirement

## Four-Week Sequence

1. Week 1: single-GPU control and TP candidate with identical workload/config.
2. Week 2: topology/NCCL evidence, per-GPU imbalance, scaling efficiency, and cost.
3. Week 3: two replicas with round-robin, least-inflight, and latency-aware routing.
4. Week 4: slow/failing replica incident and baseline-versus-speculative decoding.

PP is optional after the required work. EP, Ray/multi-node, and prefill/decode
disaggregation remain interview-level concepts.

## Implementation Status

- [x] Record exact server/router command, verified config, GPU inventory, CUDA-visible devices, NCCL environment, and `nvidia-smi topo -m`.
- [x] Preserve per-GPU telemetry and summarize memory/utilization imbalance.
- [x] Classify NCCL and distributed-worker request failures separately.
- [x] Capture speculative draft/accepted/emitted counters and acceptance rate using explicit workload counter windows.
- [x] Add one shared topology workload for single GPU, TP, and replicas.
- [x] Add replica steady/burst workload.
- [x] Add baseline/speculative decoding workload.
- [x] Add a streaming replica router with round-robin, least-inflight, latency-aware routing, bounded retry, and passive circuit breaking.
- [x] Add deterministic slow/failing-worker proxy.
- [x] Add Report 05 analyzer/template with speedup, scaling efficiency, routing, decoding, imbalance, and cost fields.
- [ ] Execute single-GPU, TP, and replica runs with identical model revision and workload.
- [ ] Execute router policies and one slow/failing-replica incident.
- [ ] Execute speculative decoding off/on if supported by the selected vLLM/model version.
- [ ] Complete Report 05 and select an architecture for a stated workload/SLO.

## Experimental Controls

Keep these identical inside each comparison:

- model repository, revision, tokenizer, and quantization
- vLLM/container, CUDA, driver, and NCCL versions
- context limit, cache/chunking/scheduler settings
- workload arrival schedule, prompt/output targets, temperature, and seed behavior
- warmup, duration, repetition count, drain guard, and aggregate GPU cost scope

Use one warmup plus at least three measured repetitions for final results. The
benchmark must record:

- actual startup/router command and passed `--expect-server-config` comparisons
- GPU count and topology matrix
- TP/PP/replica declaration
- before/after drain evidence
- before/after cumulative metric deltas

Reject conclusions when configuration evidence is missing, scheduler delay shows a
client bottleneck, or prompt/output parity materially differs.

## Core Architecture Matrix

| Deployment | Required? | Model/workload | Purpose |
| --- | --- | --- | --- |
| One GPU | Yes | Same model and `month5_topology_comparison.json` | Latency, memory, throughput, and cost baseline |
| TP across two GPUs | Yes when available | Same model/revision/workload | Measure communication overhead and memory distribution |
| Two independent replicas | Yes | Same model/revision/workload through router | Measure aggregate throughput, routing, and failure isolation |
| One H200 | Optional | Same model/revision/workload | Larger-single-GPU comparison against 2x L40S TP |
| PP | Optional | Only if fit/topology justifies it | Secondary comparison, not the center of Month 5 |

Do not compare different models and call the result a topology comparison. A 70B
fit-boundary run may be additional evidence, but the strict sharding-versus-replica
comparison uses the same checkpoint everywhere.

## Experiment 1: Single GPU Versus TP

Set the Runpod startup command before creating each pod. Example TP command:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --port 8000
```

Run the shared workload:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_topology_comparison.json \
  --server-config-label l40s_tp2 \
  --deployment-type tensor_parallel \
  --deployment-gpu-count 2 \
  --expect-server-config tensor_parallel_size=2 \
  --expect-server-config max_model_len=16384 \
  --gpu-hourly-cost-usd TOTAL_COST_OF_TWO_GPUS
```

Run the identical workload on the single-GPU control with
`--deployment-type single_gpu --deployment-gpu-count 1`, an explicit
`--tensor-parallel-size 1` startup flag, and matching
`--expect-server-config tensor_parallel_size=1`.

Measure:

- TTFT, TPOT, p95/p99, requests/sec, and output tokens/sec
- per-GPU memory and utilization imbalance
- throughput speedup and scaling efficiency
- NCCL/distributed errors
- aggregate GPU-hours and cost per million completed tokens

Interpretation target:

> TP solves model-fit or KV-capacity constraints, but it is not automatically
> faster. Frequent communication can increase TPOT and tail latency, especially on
> a weak PCIe topology.

## NCCL and Topology Notes

Know enough to explain:

- all-reduce, all-gather, reduce-scatter, and broadcast at a high level
- why TP communicates repeatedly during inference
- why same-node NVLink differs from PCIe and multi-node networking
- driver/CUDA/NCCL mismatch, process/rank mismatch, topology/network failure,
  timeout, and worker death

Preserve the topology matrix and relevant `NCCL_*` environment values in metadata.
Do not infer NCCL overhead from GPU count alone; tie it to measured TPOT/tail latency,
scaling efficiency, and topology evidence.

Use `NCCL_DEBUG=INFO` only for a separate diagnostic reproduction when a collective
or rank failure needs evidence; do not mix verbose diagnostic logging into ordinary
latency runs.

## Experiment 2: Independent Replicas and Routing

Inference replicas are not training DDP. Each replica holds the complete model, and
one request is assigned to one replica.

Use two separate Runpod endpoints or a clean image with explicit GPU assignment.
Do not start extra servers inside a pod whose PID 1 already owns its GPU.

Start the router:

```bash
python3 routing/router.py \
  --worker replica_a=http://REPLICA_A:8000 \
  --worker replica_b=http://REPLICA_B:8000 \
  --policy round_robin \
  --port 9000
```

Benchmark:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_replica_routing.json \
  --server-config-label replicas_round_robin \
  --server-launch-command \
    'python3 routing/router.py --worker replica_a=http://REPLICA_A:8000 --worker replica_b=http://REPLICA_B:8000 --policy round_robin --port 9000' \
  --deployment-type replicas \
  --deployment-gpu-count 2 \
  --expect-server-config policy=round_robin \
  --gpu-hourly-cost-usd TOTAL_COST_OF_TWO_REPLICAS
```

Repeat with `least_inflight` and `latency_aware`.

Measure achieved RPS, p95/p99, timeout/error rate, worker-attempt imbalance, GPU
balance, retries, circuit state, successful-request cost, and behavior under bursts.

When replicas are remote pods, the benchmark process cannot obtain their
`nvidia-smi` data. Collect GPU utilization/memory on each replica pod (or with a
shared DCGM/Prometheus source) and join it by run window. Do not interpret the
router host's `gpu_metrics.csv` as replica GPU balance.

Interpretation target:

> When the model fits on one GPU, replicas often provide simpler throughput scaling
> and failure isolation without TP communication, at the cost of duplicating model
> memory.

## Experiment 3: Slow/Failing Replica

Place `routing/slow_worker_proxy.py` in front of one replica and inject one condition
at a time:

- fixed latency
- every-Nth 5xx
- connection refusal/restart

Use `--max-retries 1` only for this controlled experiment. Retries must happen before
downstream response headers are sent and remain bounded.

Compare round-robin with latency-aware routing. Report retry amplification,
successful RPS, p99, timeout rate, worker imbalance, circuit opens, and recovery.

## Experiment 4: Speculative Decoding

Run `workloads/month5_speculative_decoding.json` against:

1. baseline decoding
2. the same server/model with a supported n-gram speculative configuration

Example candidate startup flag:

```bash
--speculative-config \
  '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

Record the actual `--speculative-config` command value and verify it in metadata.
Measure draft tokens, accepted tokens, acceptance rate, TPOT, p99, output throughput,
memory, and low/moderate-rate behavior.

Both decoding runs must verify shared settings such as TP size and model length. The
candidate must additionally use:

```bash
--expect-server-config \
  'speculative_config={"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

Do not assume speculative decoding improves peak throughput. Accept it only where
measured latency benefit exceeds drafting/verification overhead.

## PP and EP Boundaries

- PP: understand stage partitioning and pipeline bubbles. Run it only if a model-fit
  or non-NVLink topology question justifies the cost.
- EP: understand that MoE token routing can create expert imbalance and
  synchronization. No EP implementation is required.

## Report and Decision Rule

Generate the report with the single-GPU baseline first:

```bash
python3 scripts/analyze_month5.py \
  benchmarks/SINGLE_GPU_RUN \
  benchmarks/TP2_RUN \
  benchmarks/REPLICA_RUN \
  benchmarks/SPEC_BASELINE_RUN \
  benchmarks/SPEC_ENABLED_RUN \
  --out reports/report-05-results.md
```

Select an architecture only if it meets the workload SLO and error budget. Among
acceptable options, compare aggregate cost per completed token/request and
operational complexity. Reject “more GPUs is faster” when scaling efficiency is
poor, tail latency regresses, retries amplify load, or one worker/GPU is unstable.

## Deliverable

Report 05: Parallelism, Decoding, and Routing Trade-offs in vLLM Serving

The final interview-level statement should be:

> I compared model sharding, independent replicas, and routing for LLM inference,
> including NCCL/topology, latency, memory, failures, and cost. I used TP when fit or
> KV capacity required it and preferred replicas when the model fit per GPU and
> throughput/failure isolation were the primary goals.

## References

- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM data-parallel deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/spec_decode/)
- [vLLM parallel configuration](https://docs.vllm.ai/en/stable/api/vllm/config/parallel/)
