# Month 5 Plan: Parallelism, Decoding, and Routing Trade-offs

Goal: build hands-on credibility in multi-GPU serving and replica routing, run one bounded decoding-optimization experiment, and answer a production capacity question:

Is it better to serve a fixed workload on one larger GPU, shard one model across smaller GPUs, or replicate smaller model servers behind a router?

This is not a “multi-GPU works” demonstration. Every run must identify a fit boundary, scaling limit, routing behavior, failure mode, or cost tradeoff.

## Required Depth

| Area | Required depth |
| --- | --- |
| Tensor parallelism | Hands-on: run it, know when it is needed, measure communication cost |
| Pipeline parallelism | Run if supported/affordable; otherwise compare from documented behavior, especially on L40S without NVLink |
| Data-parallel replicas | Hands-on: run two replicas and build a simple external router |
| Greedy/sampling/max-token behavior | Hands-on and measured |
| Speculative decoding | One bounded experiment, preferably n-gram speculation first |
| Prefix/cache-aware routing | Toy implementation is sufficient |
| Ray/multi-node execution | Interview-ready conceptually; multi-node is optional |
| MoE/expert routing | Conceptual unless targeting MoE infrastructure roles |
| Prefill/decode disaggregation | Conceptual; optional experiment only after core matrix is complete |
| Custom schedulers, CUDA kernels, advanced routing research | Out of scope; do not claim ownership |

## Four-Week Sequence

1. Weeks 1-2: 1x/2x/4x L40S parallelism, model-fit, long-context, memory/utilization balance, and cost.
2. Week 3: normal decoding versus n-gram/speculative decoding, plus greedy/sampling/max-token behavior.
3. Week 4: two replicas behind a simple router, policy comparison, burst traffic, and slow/failing-worker incidents.
4. Optional after the required work: 1x H200 comparison, pipeline-parallel variant, or a small disaggregated-prefill proof of concept.

## Implementation Checklist

- [ ] Extend the launcher with `PIPELINE_PARALLEL_SIZE`, `DISTRIBUTED_EXECUTOR_BACKEND`, and `SPECULATIVE_CONFIG`.
- [ ] Record topology metadata: GPU count, TP/PP/DP sizes, visible devices, interconnect description, model revision, and server command.
- [ ] Add a topology matrix runner with warmup plus three measured repetitions per configuration.
- [ ] Capture/parse startup logs for model-fit failure, initialization time, KV-cache token capacity, and maximum-concurrency estimates.
- [ ] Preserve GPU index/rank and calculate memory/utilization imbalance rather than only aggregate maxima.
- [ ] Classify NCCL, worker/rank, distributed initialization, startup-fit, and speculative-decoding failures separately.
- [ ] Add fixed-rate, concurrency, long-context, and mixed-length Month 5 workloads shared across topologies.
- [ ] Capture speculative draft/accepted tokens and calculate acceptance rate when the server exposes them.
- [ ] Add sampling/max-token workloads with output-quality/sanity checks.
- [ ] Add `routing/` with two worker endpoints, round-robin, least-in-flight, queue approximation, retry, and circuit-breaker behavior.
- [ ] Add optional toy prefix-aware routing using a bounded prompt-prefix fingerprint; do not claim distributed KV migration.
- [ ] Add scaling efficiency, aggregate GPU-hour, router imbalance, and topology cost calculations.
- [ ] Add Report 05 and incident templates for fit, communication, context pressure, routing, and worker failure.
- [ ] Execute the required L40S matrix; execute H200/PD-disaggregation only if available and budgeted.
- [ ] Package only a selected, validated topology/router for Kubernetes/OKE-style deployment.

## Experimental Controls

Keep these identical within each controlled comparison:

- model repository and exact revision
- quantization format
- vLLM/container version and CUDA/NCCL stack
- max model length, scheduler, prefix-cache, and chunked-prefill settings
- request rate or concurrency
- prompt/output-token targets and sampling parameters
- warmup, duration, repetition count, and GPU hourly-cost assumptions

Use at least one warmup and three measured repetitions. Preserve raw run directories and startup logs. Reject a server conclusion when scheduler-delay p95 shows the client or router became the bottleneck.

## Parallelism Track

### Test Matrix

| Track | Model | Topology | Parallelism | Purpose |
| --- | --- | --- | --- | --- |
| A | Llama 3.1 8B FP8 | 1x L40S | none | Single-GPU control |
| A | Same 8B FP8 revision | 2x L40S | TP=2 | Isolate two-GPU communication overhead |
| A | Same 8B FP8 revision | 4x L40S | TP=4 | Measure scaling efficiency and over-sharding |
| A optional | Same 8B FP8 revision | 4x L40S | PP=4 or supported hybrid | Compare pipeline bubbles with TP communication |
| B | Llama 3.1 70B FP8 | 2x L40S | TP=2 | Find minimum fit and remaining KV capacity |
| B | Same 70B FP8 revision | 4x L40S | TP=4 | Measure concurrency gained from aggregate memory |
| B optional | Same 70B FP8 revision | 1x H200 | none | Larger single-GPU comparison against 4x L40S |

Track A is the strict topology comparison because the same model fits everywhere. Track B is the model-fit/capacity comparison. Do not compare 8B on one GPU with 70B on four GPUs and attribute the difference only to topology.

For a single L40S node without NVLink, explicitly compare TP and PP if the model/version supports it; current vLLM guidance notes that pipeline parallelism can reduce communication overhead on such hardware. Treat the result as hardware/version-specific.

### Shared Workload Matrix

Run the same shapes on every valid topology:

1. Fixed-rate baseline: p512/o128 at the established operating rate.
2. Concurrency sweep: c1, c8, c32, and c64.
3. Long-context sweep: p2048/o128, p8192/o128, and p15360/o128 under `MAX_MODEL_LEN=16384`.
4. Mixed-length pressure: reuse the Month 4 input pattern without changing admission policy between topologies.

### Experiment A: Model-Fit Boundary

Start the 70B FP8 checkpoint on 2x and 4x L40S. Treat BF16 or other quantization formats as separate model configurations.

Capture:

- startup success/failure and initialization time
- model/non-KV memory allocation
- vLLM-reported GPU KV-cache token capacity
- maximum-concurrency estimate at configured model length
- per-rank memory immediately after startup

Answer when the model first fits and how much usable concurrent KV capacity remains.

### Experiment B: Communication Overhead

Calculate:

- throughput speedup versus the single-GPU control
- scaling efficiency: `speedup / GPU count`
- TTFT, TPOT, p95/p99, and total-token throughput change
- aggregate GPU-hours and cost per million completed tokens
- optional 1x H200 versus 4x L40S cost/throughput difference

Determine whether increased capacity offsets collectives, synchronization, and higher aggregate cost.

### Experiment C: Long-Context and Imbalance Pressure

Track sustainable rate/concurrency, waiting requests, p99 TTFT, KV usage, preemptions, and per-GPU memory/utilization. Flag imbalance when one rank repeatedly has materially higher memory/utilization or reaches a failure boundary first.

## Decoding Track

Understand greedy versus sampling, temperature/top-p, maximum tokens/stop conditions, and prefill versus decode before interpreting speculative results.

### Experiment D: Normal vs N-Gram Speculative Decoding

Start with n-gram speculation because it does not require selecting/training a draft model. Use a supported vLLM configuration such as:

```text
{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}
```

Run the exact same deterministic workload with speculation off/on at low and moderate request rates. Compare:

- output tokens/sec and requests/sec
- TPOT and latency p95/p99
- draft/accepted tokens and acceptance rate, when available
- GPU memory overhead
- benefit for repetitive versus non-repetitive prompts
- behavior near saturation

Do not assume speculative decoding always improves throughput. Current vLLM guidance positions it primarily for inter-token-latency improvement in medium/low-QPS memory-bound workloads.

### Experiment E: Sampling and Output-Length Behavior

Run controlled cases for:

- greedy deterministic decoding
- temperature/top-p sampling with fixed seed when supported
- short versus long output caps
- early stop versus forced/target output length

Separate quality/output-sanity observations from performance. Compare actual output tokens before interpreting TPOT or tokens/sec.

## Routing Track

Run two independent vLLM replicas on separate GPU sets/ports and expose one router endpoint:

```text
routing/
  router.py
  slow_worker_proxy.py
  README.md
```

The initial implementation may use FastAPI/httpx, but it must preserve streaming, bounded timeouts, cancellation, and request IDs.

### Experiment F: Routing Policies

Implement and compare:

- round robin
- least in-flight requests
- shortest-queue approximation using bounded local/worker telemetry
- optional toy prefix-aware affinity using a prompt-prefix fingerprint

Measure per-worker request count, in-flight count, queue approximation, cache-hit evidence, latency p95/p99, throughput, timeout rate, and utilization/memory imbalance.

### Experiment G: Bursty Traffic

Send the same burst patterns through each routing policy. Determine whether the router balances arrivals or simply moves queueing from workers to the router.

### Experiment H: Slow/Failing Worker

Inject one controlled condition at a time:

- added worker latency
- worker restart/connection refusal
- repeated 5xx or timeout

Add bounded retries, passive health state, and a simple circuit breaker. Measure retry amplification, successful completion rate, p99, timeout rate, and recovery time. Never retry indefinitely or retry non-idempotent semantics without an explicit policy.

This is a toy infrastructure router, not an advanced routing algorithm or distributed KV-cache migration system.

## Conceptual/Optional Topics

### MoE and Expert Routing

Know why MoE introduces expert placement, expert parallelism, token imbalance, and synchronization concerns. Do not implement or claim expert-routing research unless the role specifically requires it.

### Prefill/Decode Disaggregation

Understand that disaggregation runs separate prefill and decode instances and transfers KV state so TTFT and inter-token latency can be tuned separately. vLLM documents this as experimental and explicitly does not present it as a throughput optimization.

Optional hypothesis for a 4x L40S node:

- GPUs 0-1: prefill instance with TP=2
- GPUs 2-3: decode instance with TP=2
- same 70B FP8 revision if it fits each pair
- compare against a non-disaggregated 4-GPU control

Do not state that 4x L40S beats 2x A100/H200 without measured identical-model/workload evidence. Record PCIe/NVLink topology and KV-transfer overhead. Use a clean base image when a preconfigured image would consume all GPUs before the intended topology is launched.

### Ray and Multi-Node

Know that current vLLM defaults to native multiprocessing for supported single-node execution and uses Ray for multi-node execution/placement. Multi-node execution is optional for this project.

## Required Metrics and Artifacts

- TTFT, TPOT, and end-to-end latency p50/p95/p99
- achieved request rate, requests/sec, and tokens/sec
- scheduler/router delay and waiting/running requests
- KV-cache utilization, prefix-cache hit rate, and preemptions
- speculative draft/accepted-token metrics and acceptance rate
- per-GPU/rank memory and utilization, not only averages
- per-replica request/in-flight/error/retry/circuit state
- startup capacity/concurrency estimates
- timeout, OOM, rejection, server, worker, NCCL, and routing error counts
- aggregate GPU-hours and cost per million completed tokens
- exact server/router commands, model revision, topology, and run IDs

## Decision Rule

Select an architecture only if it meets the workload SLO and error budget. Among acceptable options, compare cost per completed request/token and operational complexity. A faster design is not better when scaling efficiency is poor, tail latency fails, retries amplify load, one rank/replica is unstable, or aggregate cost is materially higher.

## Kubernetes/OKE-Style Packaging

After selecting a validated topology/router, add:

- container images and Kubernetes workload controller
- Service/Ingress or router Service
- GPU resources matching TP/PP/replica layout
- readiness/liveness probes and bounded termination/drain behavior
- Prometheus scraping for workers and router
- topology-aware scheduling and GPU/node constraints
- optional Helm values for model, replica/GPU count, TP/PP, context, memory, decoding, and routing policy

Do not package an unvalidated topology. Kubernetes should reproduce the bare-metal server/router commands and metrics rather than introduce another experiment simultaneously.

## Deliverables

- Report 05: Parallelism, Decoding, and Routing Trade-offs in vLLM Serving
- Model-fit/startup-capacity table
- Repeated topology table with speedup, scaling efficiency, and cost
- Normal/speculative decoding comparison
- Router policy and slow/failing-worker incident comparison
- Per-GPU and per-replica imbalance timeline
- Selected production architecture and rejected alternatives
- Kubernetes/OKE-style manifests for the selected architecture

## Acceptance Checks

- The same 8B model/workload runs on 1x/2x/4x L40S for strict topology comparison.
- The same 70B FP8 model/workload runs on valid 2x/4x L40S and optional H200 configurations.
- Every reported point has warmup plus at least three measured repetitions.
- Startup KV capacity/maximum-concurrency estimates and per-rank telemetry are preserved.
- Speculative decoding uses the same target model/workload and reports acceptance evidence when available.
- Router policies receive identical arrivals; router delay and retry load are measured separately.
- Slow/failing-worker tests use bounded retries/circuit behavior and show recovery.
- MoE and PD disaggregation claims remain conceptual unless measured.
- H200/PD experiments occur only after required L40S, decoding, and routing work is ready.
- The final recommendation names the workload/SLO and answers when to use one large GPU, model sharding, or replicas.

## References

- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM data-parallel deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)
- [vLLM speculative decoding](https://docs.vllm.ai/en/stable/features/spec_decode/)
- [vLLM disaggregated prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
