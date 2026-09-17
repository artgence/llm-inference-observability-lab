# LLM Inference Performance & Observability

A reproducible performance-engineering toolkit for **benchmarking, diagnosing, and comparing GPU-backed LLM serving systems**. The project connects request-level latency to server queues, KV-cache pressure, GPU utilization, and deployment cost so that serving decisions can be evaluated against a workload and a service-level objective—not throughput alone.

**Stack:** Python · vLLM · PyTorch · Hugging Face Transformers · CUDA · Prometheus · Grafana · NVIDIA DCGM · Docker Compose · Runpod

## What this project delivers

- **A streaming benchmark harness** with fixed-concurrency, fixed-arrival-rate, and burst workloads; request-level latency, token usage, failure classification, and cost summaries.
- **A correlated observability pipeline** connecting client events, vLLM metrics, GPU samples, and experiment metadata, with provisioned dashboards and alert rules.
- **A replica-routing implementation** with round-robin, least-inflight, and latency-aware policies, bounded retries, passive circuit breaking, and per-worker request provenance.
- **A controlled experiment framework** for evaluating serving configuration, single-GPU versus tensor-parallel versus replicated deployments, and speculative decoding.
- **A PyTorch profiling companion** for inspecting prefill/decode execution, CUDA memory, attention shapes, and the differences between in-process generation and a serving engine.

The implementation focuses on three connected concerns: **measurement validity, bottleneck diagnosis, and deployment trade-offs**.

## System architecture

```text
JSON workload definitions -> Streaming benchmark client
                                      |
                           +----------+----------+
                           |                     |
                      Direct vLLM          Replica router
                    (single GPU / TP)       /           \
                                       Replica A     Replica B

Request records + vLLM/router metrics + per-host GPU samples
                           |
                 Run-correlated artifacts
                           |
           Capacity, configuration, and incident analysis
```

The client drives load and records user-visible behavior. Server and router metrics explain queueing and worker activity. GPU collectors provide resource context; remote replicas require collection on each serving host. Analyzers read run summaries and metadata to produce comparison tables and incident notes. Prometheus/Grafana provide an optional live view of the same system.

## Engineering workflow

### 1. Establish a trustworthy performance baseline

Capacity measurements start with a controlled workload and a known server configuration. The [benchmark client](scripts/benchmark_vllm.py) supports both closed-loop concurrency and open-loop arrivals, including bursts, to distinguish “how many requests can run at once?” from “what happens when traffic arrives at a fixed rate?”

It records time to first token (TTFT), time per output token (TPOT), end-to-end latency, throughput, errors, and client scheduling delay. Configuration expectations, local GPU inventory checks, and an idle-queue drain guard help prevent accidentally comparing different deployments or overlapping workload stages.

**Purpose:** establish the capacity/latency boundary before attempting optimization. See the [baseline and workload sensitivity experiments](docs/runbook.md#experiment-reference).

### 2. Explain bottlenecks across the serving stack

The observability path links request symptoms to engine and hardware state: waiting requests, KV-cache usage, cache hits, preemptions, GPU allocation, and utilization. [Dashboard configuration](observability/grafana/dashboards/vllm-observability.json) and [alert rules](observability/prometheus/alerts.yml) cover queue growth, slow first-token latency, metrics availability, and request failures.

For deeper execution analysis, the [PyTorch tools](pytorch_baseline/README.md) separate prefill and decode, capture CUDA memory statistics and profiler traces, and preserve prompt fingerprints for comparison with vLLM. Incident workloads exercise traffic bursts, long prompts, and memory pressure; the [incident analyzer](scripts/diagnose_incident.py) produces evidence-linked diagnostic notes with provisional bottleneck hypotheses.

**Purpose:** distinguish client limitations, server queueing, model execution, and memory pressure before choosing a mitigation.

### 3. Evaluate single-GPU serving efficiency

Controlled workload pairs cover prefix caching, chunked prefill, batching limits, context capacity, and client-side admission policies. The [configuration analyzer](scripts/analyze_month4.py) compares latency, queue depth, cache behavior, errors, and cost while retaining configuration evidence.

Speculative decoding is evaluated as a separate off/on experiment on the same single-GPU allocation. The [test procedure](docs/runbook.md#speculative-decoding-step-by-step) controls output length and examines draft acceptance alongside end-user latency and throughput. It does not assume that higher acceptance automatically means a faster service.

**Purpose:** determine which change improves the target workload, where the benefit ends, and what capacity or reliability trade-off it introduces.

### 4. Compare scaling and resilience strategies

The deployment comparison holds the model and workload constant across one GPU, tensor parallelism across two GPUs, and two independent one-GPU replicas. This separates the question of model fit and inter-GPU communication from request-level throughput scaling.

The [streaming router](routing/router.py) adds three routing policies, bounded pre-response retries, and passive circuit breaking. It records selected worker, attempt number, retry history, per-worker latency, active requests, queue depth when available, and telemetry completeness. [Integration tests](tests/test_router.py) check that the first SSE event is forwarded before generation finishes and that retry identity is preserved.

**Purpose:** compare latency, scaling efficiency, failure behavior, and aggregate GPU cost with evidence that both workers actually handled traffic. See the [replica runbook](docs/runbook.md#two-replicas-and-routing) for remote telemetry and the unfinished fault-proxy path.

### 5. Turn measurements into reviewable decisions

Each run preserves request records, workload-correlated metrics, local GPU samples, and metadata. The [architecture analyzer](scripts/analyze_month5.py) calculates matched-workload speedup, scaling efficiency, and cost comparisons; report templates structure the final explanation around the workload, service-level objective, bottleneck, and rejected alternatives.

**Purpose:** make a recommendation traceable to its configuration, workload, measurement window, and limitations. Reports are review artifacts, not an automatic guarantee that an experiment was fair.

## Measurement design and safeguards

Several design choices address ways an otherwise plausible benchmark can be misleading:

| Risk | Control or explicit boundary |
| --- | --- |
| Benchmark label differs from the running server | Capture startup/configuration evidence and check expected settings. A manually supplied command remains supplied evidence, not independent verification. |
| One workload overlaps the next, or counters include earlier traffic | Require stable idle queues and use named before/after counter windows. |
| Load generation becomes the bottleneck | Record scheduled versus actual request start times and client scheduler delay. |
| Different prompt/output lengths make comparisons unequal | Check observed token usage and prefix-test prompt parity; use paired output policies. `max_tokens` alone is only a cap. |
| Buffering makes TTFT look like total latency | Preserve SSE delivery and test streaming behavior. Chunk timing is not necessarily token timing. |
| Local telemetry is mistaken for a whole replica deployment | Distinguish declared deployment GPU count from local inventory; collect remote GPUs separately. |
| Missing queue metrics appear as zero load | Track per-worker completeness and omit aggregate queue gauges when worker data is incomplete. |

The detailed [measurement rules](docs/runbook.md#configuration-and-fair-comparisons) cover warmup, repetitions, revisions, cost scope, and interpretation. These controls support trustworthy measurements; they do not replace checking raw evidence.

## Evidence and current scope

The repository contains the benchmark/router implementation, **17 workload definitions**, profiling and analysis tools, observability configuration, and regression tests. Tests cover configuration checks, drain behavior, counter windows, prompt parity, telemetry scope, routing policies, streaming, retries, and scaling calculations.

The checked-in [reports](reports/) and [incident notes](incidents/) are currently templates awaiting completed GPU-backed evidence. This README therefore makes no numerical speedup, cost-reduction, or production-reliability claim. The experiment setup targets Llama 3.1 8B FP8 on L40S and H100 SXM hardware; matched comparisons must keep the hardware/model scope explicit.

Known boundaries:

- The router is an experimental serving component, not a production gateway. Its health view uses passive routing/circuit state rather than a complete active health-check system.
- The fault-injection proxy still buffers streams and does not expose the queue metrics needed by the drain guard; end-to-end fault-performance validation remains unfinished.
- Remote GPU samples are not automatically merged, and comparison scripts do not automatically compute repetition medians or establish every fairness condition.
- The project builds on vLLM/PyTorch; it does not implement a new inference engine, CUDA kernel, or distributed training system.

## Run the project

### Validate locally without a GPU

The benchmark/router and basic analyzers use the Python standard library. From the repository root:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month1_baseline.json \
  --dry-run

python3 -m unittest discover -s tests -v
python3 -m unittest discover -s pytorch_baseline -p 'test_*.py' -v
```

Dry-run checks workload declarations only. Router integration tests need permission to bind localhost ports; they do not require a GPU or a live vLLM server.

### Benchmark a running vLLM server

First follow the [environment and server setup](docs/runbook.md#runpod-setup-and-first-benchmark). In a separate terminal on the serving host, from the repository root:

```bash
export MODEL_ID=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8
export SERVED_MODEL_NAME="$MODEL_ID"
export VLLM_BASE_URL=http://127.0.0.1:8000
export OPENAI_API_KEY=EMPTY  # Replace if authentication is enabled.

python3 scripts/openai_smoke_test.py
python3 scripts/benchmark_vllm.py \
  --workload workloads/month1_baseline.json \
  --run-id baseline_r1 \
  --server-config-label single_gpu_baseline \
  --deployment-type single_gpu \
  --deployment-gpu-count 1 \
  --expect-local-gpu-count 1 \
  --expect-server-config tensor_parallel_size=1 \
  --expect-server-config max_model_len=16384
```

This example assumes a one-GPU server started explicitly with TP=1 and a 16,384-token context limit. Use a new run ID each time. Expectations check configuration; they do not start or reconfigure the server. The smoke script checks generation, not incremental streaming—use the runbook's separate streaming check before interpreting TTFT.

Each run writes `metadata.json`, `requests.jsonl`, `summary.csv`, `summary.md`, server/router metric snapshots, and local GPU samples when available. Generated artifacts under `benchmarks/` are ignored by Git; preserve and review them before publishing conclusions.

## Code and documentation map

| Component | Entry point |
| --- | --- |
| Workload generation, streaming measurements, and validity checks | [scripts/benchmark_vllm.py](scripts/benchmark_vllm.py) |
| Replica selection, SSE forwarding, retries, and worker metrics | [routing/router.py](routing/router.py) |
| In-process generation and CUDA profiling | [pytorch_baseline/](pytorch_baseline/) |
| Metrics collection, dashboards, and alerts | [observability/](observability/), [GPU collector](scripts/collect_gpu_metrics.py) |
| Capacity, configuration, architecture, and incident analysis | [scripts/](scripts/) |
| Controlled experiment inputs | [workloads/](workloads/) |
| Regression and local HTTP integration tests | [tests/](tests/) |
| Setup, benchmark commands, telemetry interpretation, and troubleshooting | [Operating runbook](docs/runbook.md) |

Existing filenames retain their original experiment identifiers for compatibility; the project is organized here by the engineering questions they answer.
