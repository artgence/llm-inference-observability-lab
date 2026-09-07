# LLM Inference Observability Lab

A hands-on lab for understanding **how LLM serving behaves under load**: latency, throughput, queueing, GPU memory, failures, and cost. It combines a streaming vLLM benchmark client, GPU/server telemetry, controlled workloads, a small replica router, and report templates.

The central question is not just “how many tokens per second?” It is **which serving configuration meets a workload's latency and reliability requirements, with evidence that makes the comparison trustworthy?**

The initial baseline uses `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` on an L40S. Month 5 compares H100 SXM deployments: one GPU, TP=2, two independent replicas, and speculative decoding on the single-GPU control. These are experiment targets, not performance claims. Reports containing `TBD` or “awaiting” are templates, not completed measurements.

## Contents

- [Experiment map](#experiment-map)
- [Runpod setup and first benchmark](#runpod-setup-and-first-benchmark)
- [Configuration and fair comparisons](#configuration-and-fair-comparisons)
- [Single GPU versus TP=2](#single-gpu-versus-tp2)
- [Two replicas and routing](#two-replicas-and-routing)
- [Speculative decoding: step by step](#speculative-decoding-step-by-step)
- [Results and observability](#results-and-observability)
- [Troubleshooting](#troubleshooting)
- [Local validation and project layout](#local-validation-and-project-layout)

## Experiment map

| Stage | Question | Workloads and guide |
| --- | --- | --- |
| Month 1 | What is the baseline, and where does throughput saturate? | `month1_baseline.json`, `month1_open_loop.json`; [plan](notes/month-01-plan.md) |
| Month 2 | How do prompt length, output length, and bursts change capacity? | `month2_*_sweep.json`; [plan](notes/month-02-plan.md) |
| PyTorch baseline | What does in-process inference reveal about prefill, decode, and memory? | [Setup, profiling, and comparison](pytorch_baseline/README.md) |
| Month 3 | Can telemetry explain bursts, long-prompt storms, and memory pressure? | `month3_*_incident.json`; [plan](notes/month-03-plan.md) |
| Month 4 | What do prefix caching, batching, context limits, and admission control change? | `month4_*.json`; [plan](notes/month-04-plan.md) |
| Month 5 | Should the same model use one GPU, TP=2, or replicas? Does speculation help? | `month5_*.json`; [plan](notes/month-05-plan.md) |

Workload paths above are under `workloads/`. Pipeline parallelism is optional; expert parallelism is conceptual. Kubernetes, Ray, multi-node TP, and distributed KV-cache routing are not prerequisites.

## Runpod setup and first benchmark

Commands below run in **Bash inside the GPU Pod**, from the repository root unless stated otherwise. A local CPU machine can validate workloads and run lightweight tests, but cannot produce GPU-backed serving results.

### 1. Choose the environment

1. Create a GPU Pod with a vLLM-capable image. Use one H100 SXM for the Month 5 control, two H100 SXMs in one Pod for TP=2, or two one-H100-SXM Pods for replicas.
2. Keep the image tag/digest, model revision, tokenizer, and runtime identical across comparison arms. The Runpod sessions motivating this guide reported vLLM `0.25.0`; this is a reference version, not a claim that it is the latest or that every image is compatible.
3. Allocate persistent storage for the repository, model cache, and results. Confirm its actual mount path. Runpod normally mounts a volume at `/workspace`; `/vllm-workspace` is not automatically persistent. Container-disk data can be lost on stop/edit, and a Pod volume is deleted on termination. Back up important results outside the Pod. See [Runpod storage](https://docs.runpod.io/pods/storage/types).
4. Supply `HF_TOKEN` through a secret/environment setting if model access requires it, and complete any required Hugging Face license/access steps. Never commit tokens or paste them into reports.
5. For replicas, enable Global Networking on both supported Pods. Prefer the same data center for controlled comparisons. Private services must listen on a reachable interface such as `0.0.0.0`. See [Runpod networking](https://docs.runpod.io/pods/networking).

These examples are for a controlled private lab. Do not expose unauthenticated generation, metrics, router, or dashboard ports publicly. `OPENAI_API_KEY=EMPTY` is a client placeholder, not authentication protection.

### 2. Open a terminal and check the Pod

For a new checkout on a confirmed `/workspace` volume:

```bash
cd /workspace
git clone https://github.com/artgence/llm-inference-observability-lab.git
cd llm-inference-observability-lab

python3 --version
python3 -c 'import vllm; print(vllm.__version__)'
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
git rev-parse HEAD
```

If the checkout already exists, enter it instead of cloning again. For the earlier container layout, use `cd /vllm-workspace/llm-inference-observability-lab`; verify storage persistence before relying on it. Preserve local edits when updating a checkout.

The benchmark, router, and basic analyzers use the Python standard library. They do not require the separate PyTorch profiling environment. If `import vllm` fails, prepare a compatible, pinned vLLM environment before continuing; do not upgrade one comparison arm independently.

### 3. Start or verify the single-GPU server — terminal 1

First check whether the image already starts vLLM:

```bash
ps -eo pid,args | grep '[v]llm'
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8000/health
```

If vLLM already owns the GPU/port, use that server and inspect its launch configuration. If it is the container's PID 1, change the Pod/template startup configuration to change flags; do not launch a second server or kill PID 1 from a benchmark shell. Save results before any Pod restart/edit.

On a clean interactive environment with no existing server, start:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384
```

Keep this terminal open, or use your chosen process supervisor. Wait for model loading and readiness. For reproducible results, also pin the model/tokenizer revision and record the full effective command and image digest.

There is **no `scripts/start_vllm_server.sh` in this repository**. `.env.example` is a reference, not a launcher; `.env` is not loaded automatically. Variables such as `TENSOR_PARALLEL_SIZE` and `MAX_MODEL_LEN` do not change a running server. Set the corresponding vLLM startup flags explicitly.

### 4. Check the endpoint and streaming — terminal 2, same Pod

Enter the same checkout, then set these in each new benchmark shell:

```bash
export MODEL_ID=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8
export SERVED_MODEL_NAME="$MODEL_ID"
export VLLM_BASE_URL=http://127.0.0.1:8000
export OPENAI_API_KEY=EMPTY

curl -sS -o /dev/null -w 'HTTP %{http_code}\n' "$VLLM_BASE_URL/health"
curl -sS "$VLLM_BASE_URL/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" | python3 -m json.tool
python3 scripts/openai_smoke_test.py
```

Use the actual served name from `/v1/models` if the server has a custom alias. Use the configured API key instead of `EMPTY` when authentication is enabled.

The smoke script is non-streaming. Separately verify that content arrives incrementally:

```bash
curl -sS -N --max-time 120 "$VLLM_BASE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Explain how an inference server processes a request."}],
    "max_tokens": 128,
    "temperature": 0,
    "ignore_eos": true,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Change the JSON model if using an alias. `curl -N` disables curl's buffering; it cannot undo an upstream proxy's buffering. Do not pipe SSE through `python3 -m json.tool`: it is multiple `data:` events, not one JSON document. An empty vLLM `/health` body with HTTP 200 is normal.

### 5. Validate and run a first workload

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month1_baseline.json \
  --dry-run

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

`--dry-run` validates declarations without traffic; it does **not** verify a live server or GPU. Real runs check requested expectations and wait for idle running/waiting queues before and after each workload. The drain guard defaults to a 120-second timeout. Keep it enabled for performance measurements.

Each run ID must be new. Use `baseline_r2`, a timestamp, or omit `--run-id` for an automatically generated ID. A failed attempt may already have created its result directory; inspect it and use a new ID for the retry.

## Configuration and fair comparisons

| Setting | Meaning |
| --- | --- |
| `VLLM_BASE_URL` | Endpoint root, e.g. `http://127.0.0.1:8000` or router port `9000`; no `/v1` suffix. Overrides the workload URL. |
| `SERVED_MODEL_NAME` / `MODEL_ID` | Client model name; precedence is served name, model ID, then workload model. |
| `--server-config-label` | Human-readable experiment label only. |
| `--server-launch-command` | Records an already-running command and overrides local discovery. Never starts or changes a server. |
| `--expect-server-config KEY=VALUE` | Checks captured launch arguments/config metrics. Does not configure vLLM. A supplied command is supplied evidence, not independent live verification. |
| `--deployment-gpu-count` | Declared deployment/accounting count, including remote replicas. Not measured inventory. |
| `--expect-local-gpu-count` | Checks the benchmark host's `nvidia-smi` inventory only. |
| `--expect-gpu-name H100` | Checks local names for `H100`; inspect full model/topology to establish SXM hardware. |
| `--gpu-hourly-cost-usd` / `GPU_HOURLY_COST_USD` | Aggregate hourly GPU cost attributed to the endpoint, not a per-GPU price to multiply automatically. |

For a control pinned to one GPU on a two-GPU host, deployment count is `1` while local inventory can be `2`. `CUDA_VISIBLE_DEVICES=0` limits the serving process; it does not necessarily restrict `nvidia-smi` inventory. Local checks cannot validate remote replicas.

Before drawing a conclusion:

- Keep model revision, tokenizer/chat template, quantization, runtime, context/cache settings, prompt data, sampling, and output policy identical. Change one factor at a time.
- Use one excluded warmup and at least three measured repetitions per configuration. Record cache state and restart boundaries; do not run competing benchmarks on the same server.
- Inspect actual prompt/output usage and `token_count_source`. Synthetic prompt targets do not prove exact tokenizer parity.
- Treat `max_tokens` as a cap. Existing workloads can stop at EOS; `output_tokens_target` is a label, not enforcement. For equal-output comparisons, create a paired workload copy and add `"extra_body": {"ignore_eos": true}` to **each run object** in both arms. Remove other early-stop settings and verify actual output counts; context limits and failures can still truncate requests.
- Do not override `stream` or `stream_options` through `extra_body` during streaming measurements. Extra fields are merged directly into the request body.
- Check achieved arrival rate and `scheduler_delay_p95_s`. Scheduling delay or a `max_in_flight` ceiling can make the client the bottleneck. Fixed-rate tests below capacity do not establish maximum throughput.
- Report median results and spread across repetitions. Increase duration/request counts before relying on tail percentiles; a short smoke workload is not a capacity study.
- Record cost scope explicitly: one attributed GPU versus a whole rented two-GPU Pod are different accounting choices. Use the actual rental rate, not a guessed price.

Earlier TP runs stopped at different average output lengths (roughly 119–121 versus 126 tokens/request). Those observations are not an equal-work speedup: token throughput, request throughput, and latency all depend on generated work. Preserve them as exploratory results and rerun paired output policies for architecture conclusions.

## Single GPU versus TP=2

Use `workloads/month5_topology_comparison.json` for **all three architecture arms**, including replicas. It contains short-prompt fixed-rate, concurrency-32, and long-prompt cases.

On the one-H100 control, with the single-GPU server above:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_topology_comparison.json \
  --run-id m5_single_r1 \
  --server-config-label h100_sxm_single \
  --deployment-type single_gpu \
  --deployment-gpu-count 1 \
  --expect-gpu-name H100 \
  --expect-local-gpu-count 1 \
  --expect-server-config tensor_parallel_size=1 \
  --expect-server-config max_model_len=16384
```

On a **two-H100 Pod**, configure the server at startup with:

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 16384
```

After readiness, in a second terminal on that Pod:

```bash
VLLM_BASE_URL=http://127.0.0.1:8000 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_topology_comparison.json \
  --run-id m5_tp2_r1 \
  --server-config-label h100_sxm_tp2 \
  --deployment-type tensor_parallel \
  --deployment-gpu-count 2 \
  --expect-gpu-name H100 \
  --expect-local-gpu-count 2 \
  --expect-server-config tensor_parallel_size=2 \
  --expect-server-config max_model_len=16384
```

Set aggregate cost for each arm if reporting economics. Repeat after a separate warmup. A TP=1 startup must have an expectation of `tensor_parallel_size=1`; changing the expectation to `2` cannot create TP=2. Inspect topology/NCCL evidence before attributing differences to communication overhead.

## Two replicas and routing

Each replica holds a full model copy and starts with `--tensor-parallel-size 1`. This is request-level replication, not one model sharded across Pods. Start and verify both servers using the single-GPU steps.

### 1. Verify workers from the router Pod

Run the router and benchmark together on Pod A for the simplest layout. Replace both placeholder Pod IDs:

```bash
export REPLICA_A=http://REPLICA_A_POD_ID.runpod.internal:8000
export REPLICA_B=http://REPLICA_B_POD_ID.runpod.internal:8000

getent hosts REPLICA_A_POD_ID.runpod.internal
getent hosts REPLICA_B_POD_ID.runpod.internal

for worker in "$REPLICA_A" "$REPLICA_B"; do
  curl -sS -o /dev/null -w 'HTTP %{http_code}\n' "$worker/health"
  curl -sS "$worker/v1/models" \
    -H "Authorization: Bearer $OPENAI_API_KEY" | python3 -m json.tool
  curl -sS "$worker/metrics" | grep -E '^vllm:num_requests_(running|waiting)'
done

VLLM_BASE_URL="$REPLICA_A" python3 scripts/openai_smoke_test.py
VLLM_BASE_URL="$REPLICA_B" python3 scripts/openai_smoke_test.py
```

A health request alone does not prove generation works. Private DNS needs networking configured on both Pods; `getent hosts google.com` only tests public DNS. Do not replace Docker's resolver just because a private name is missing. Private Pod networking is not same-host NVLink: document the network path when comparing replicas and TP. See [Runpod private networking](https://docs.runpod.io/pods/networking).

### 2. Keep the router running on Pod A

From the checkout on A, with no existing router occupying port 9000:

```bash
export ROUTER_COMMAND="python3 routing/router.py --worker replica_a=$REPLICA_A --worker replica_b=$REPLICA_B --policy round_robin --host 127.0.0.1 --port 9000"

nohup python3 routing/router.py \
  --worker "replica_a=$REPLICA_A" \
  --worker "replica_b=$REPLICA_B" \
  --policy round_robin \
  --host 127.0.0.1 \
  --port 9000 \
  > router-round-robin.log 2>&1 &
ROUTER_PID=$!
echo "$ROUTER_PID"

curl -sS http://127.0.0.1:9000/health | python3 -m json.tool
curl -sS http://127.0.0.1:9000/metrics
```

Check the log if startup fails; retry the checks once it is listening. `127.0.0.1:9000` exists only on this Pod, not on B. The localhost binding is intentional because this example colocates the benchmark.

Router `/health` describes routing/circuit state; it does not actively test every worker. Fresh zero counters are not proof that both work. Router `/metrics` scrapes each worker's `/metrics` and exports aggregate running/waiting gauges only when **all workers provide both values**. Missing queue data is unknown, not zero.

### 3. Benchmark through port 9000

In the same shell on A:

```bash
VLLM_BASE_URL=http://127.0.0.1:9000 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_topology_comparison.json \
  --run-id m5_replicas_round_robin_r1 \
  --server-config-label h100_sxm_replicas_round_robin \
  --server-launch-command "$ROUTER_COMMAND" \
  --deployment-type replicas \
  --deployment-gpu-count 2 \
  --expect-server-config policy=round_robin
```

The explicit URL is essential: the topology workload defaults to port 8000. The supplied command documents the router but does not independently verify its live policy; save `/health` and startup logs too. Verify each worker's GPU and TP=1 configuration separately.

For routing-policy/burst tests, use `workloads/month5_replica_routing.json` and a new run ID. Repeat with `least_inflight` and `latency_aware`, changing the **running router command** and its recorded label/expectation. Stop only the router you started (`kill "$ROUTER_PID"` in that shell), confirm shutdown, and restart with the next policy. Keep workers and workload unchanged.

### 4. Validate worker identities and GPU evidence

Do not accept aggregate “zero failures” as the whole result. Current router/client evidence includes:

| Evidence | Where to inspect |
| --- | --- |
| Selected worker and attempt number | `requests.jsonl`: `router_selected_worker`, `router_attempt_number` |
| Retry path and request ID | `router_attempt_history`, `router_request_id`; router attempt logs |
| Worker requests, successes, errors | `summary.csv`: `router_worker_request_counts`, `router_worker_success_counts`, `router_worker_error_counts` |
| Worker latency | `router_worker_latency_seconds_avg`, `router_worker_latency_ewma_seconds_end` |
| Active requests and backend load | `router_worker_active_requests_max_observed`, `router_worker_engine_running_requests_max_observed` |
| Worker queue depth | `router_worker_queue_depth_max_observed` |
| Health and telemetry completeness | `router_worker_health_states`, `router_worker_metrics_complete`, `router_upstream_metrics_complete` |
| Time series and worker labels | `vllm_metrics.jsonl`; `llm_router_worker_*{worker="..."}` metrics |

Worker request counts count **attempts**, so retries can make their sum exceed client requests. Worker latency is full router-to-worker attempt duration, including failed/cancelled attempts, not backend-only inference time. Sampled maxima can miss brief peaks. Compact Markdown reports omit some detailed maps; inspect CSV/raw metrics.

The benchmark's `gpu_metrics.csv` samples **only GPUs visible on its host**. A two-replica result with `gpu_count_observed=1` neither disproves traffic to B nor proves two-GPU utilization. Collect telemetry separately on **each serving Pod** during the same run window:

```bash
mkdir -p benchmarks/worker-telemetry
WORKER=replica_a  # Set replica_b on Pod B.
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv \
  > "benchmarks/worker-telemetry/${WORKER}-inventory.csv"
python3 scripts/collect_gpu_metrics.py \
  --output "benchmarks/worker-telemetry/${WORKER}-gpu.csv" \
  --interval 2 \
  --duration 900
```

Run collectors in separate terminals before benchmarking; choose a duration covering the run. Preserve Pod ID, worker name, GPU UUID, timestamps, and run IDs. Join samples by worker and workload window, never just GPU index: both Pods can have GPU `0`. Remote GPU files are **not automatically merged** into the summary/analyzer. Use distinct filenames per collection to preserve earlier samples.

### Fault-injection limitation

`routing/slow_worker_proxy.py` provides deterministic delays and HTTP failures, but currently buffers forwarded responses and its `/metrics` lacks upstream running/waiting gauges. Putting it in the worker path invalidates streaming timing and prevents the strict drain guard from establishing idle queues.

Treat end-to-end fault-proxy performance tests as unfinished until streaming and queue telemetry are corrected and tested. **Do not disable the drain guard to make this test pass.** In a validated fault setup, test delay and failure separately before combining them. Router retries default to zero; `--max-retries 1` permits a bounded eligible retry before response headers are sent. See [router notes](routing/README.md), subject to this limitation.

## Speculative decoding: step by step

Test **directly against one H100 on port 8000**, without the router/public proxy. Start with n-gram speculation: proposals come from repeated token sequences, without a second draft model. This tests one method, not every draft-model/EAGLE/MTP approach. Configuration is version-sensitive; see the [vLLM n-gram guide](https://docs.vllm.ai/en/v0.25.0/features/speculative_decoding/n_gram/) and [v0.25.0 configuration](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/config/speculative.py).

### 1. Prepare a paired workload

`month5_speculative_decoding.json` has 512-token prompt targets and a 256-token output cap: 40 requests at 2 RPS, then 120 at 12 RPS. It does not suppress EOS.

```bash
cp workloads/month5_speculative_decoding.json \
  workloads/month5_speculative_decoding_fixed_output.json
```

Edit the copy, adding `"extra_body": {"ignore_eos": true}` inside **both objects in `runs`**, alongside `max_tokens` and `temperature`. Keep everything else unchanged. Validate:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month5_speculative_decoding_fixed_output.json \
  --dry-run
```

Both cases should list `ignore_eos` in `extra_body_keys`. Preserve this workload with the results; use the same file for off/on.

### 2. Measure speculation off

Use the single-GPU startup command, with **no `--speculative-config`**. Check actual startup configuration, health, and streaming. Run one excluded warmup and three measured repetitions:

```bash
export VLLM_BASE_URL=http://127.0.0.1:8000
export MODEL_ID=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8
export SERVED_MODEL_NAME="$MODEL_ID"
export OPENAI_API_KEY=EMPTY

for repetition in warmup r1 r2 r3; do
  python3 scripts/benchmark_vllm.py \
    --workload workloads/month5_speculative_decoding_fixed_output.json \
    --run-id "m5_spec_off_${repetition}" \
    --server-config-label h100_sxm_spec_off \
    --deployment-type single_gpu \
    --deployment-gpu-count 1 \
    --expect-gpu-name H100 \
    --expect-local-gpu-count 1 \
    --expect-server-config tensor_parallel_size=1 \
    --expect-server-config max_model_len=16384 || break
done
```

Use the real API key when required. Set the same one-GPU hourly cost in both arms if comparing economics. Inspect any failure; do not average incomplete runs.

### 3. Restart the same allocation with n-gram speculation

Save results and change only the speculative configuration. Stop a manually launched server with Ctrl-C in its terminal; for auto-starting images, change Pod startup settings and account for storage/restart behavior.

```bash
vllm serve neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

Keep pinned revisions and other baseline flags identical. Record an unsupported configuration as unsupported; do not quietly change model, quantization, or runtime in just this arm.

### 4. Verify activation and repeat

In the benchmark terminal, re-enter the checkout and restore step 2's client environment after any restart. Define:

```bash
export SPEC_CONFIG='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8000/health
python3 scripts/openai_smoke_test.py
curl -sS http://127.0.0.1:8000/metrics | grep 'spec_decode'

for repetition in warmup r1 r2 r3; do
  python3 scripts/benchmark_vllm.py \
    --workload workloads/month5_speculative_decoding_fixed_output.json \
    --run-id "m5_spec_ngram4_${repetition}" \
    --server-config-label h100_sxm_spec_ngram4 \
    --deployment-type single_gpu \
    --deployment-gpu-count 1 \
    --expect-gpu-name H100 \
    --expect-local-gpu-count 1 \
    --expect-server-config tensor_parallel_size=1 \
    --expect-server-config max_model_len=16384 \
    --expect-server-config "speculative_config=$SPEC_CONFIG" || break
done
```

Expected JSON should match the startup argument's serialization; the comparator does not canonicalize JSON objects. Inspect metadata/startup logs if discovery fails. Do not fabricate a launch-command override to force a match.

### 5. Decide from acceptance and end-user performance

Compare actual output lengths, TPOT, TTFT, p95/p99 latency, successful RPS, output tokens/sec, errors, GPU memory/utilization, and cost across matched repetitions. A benefit at 2 RPS need not survive higher load; these short cases are a first check, not a saturation study.

Use **per-workload before/after counter deltas**, excluding warmup. Relevant vLLM 0.25.0 counters include `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, and `vllm:spec_decode_num_accepted_tokens_total`:

```text
acceptance rate = delta(accepted tokens) / delta(draft tokens)
mean acceptance length = 1 + delta(accepted tokens) / delta(drafts)
```

Treat zero denominators as unavailable. The harness summarizes draft/accepted tokens and acceptance; the draft-count counter is in raw snapshots. The referenced runtime does not expose the emitted-token counter the harness optionally recognizes; an empty emitted-token field alone does not mean speculation is off. See [vLLM speculative metrics](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/v1/spec_decode/metrics.py).

N-gram proposals may be rare or poorly accepted for this synthetic workload. “No improvement” is a valid result. If adding repetition-friendly prompts, keep them as a separate off/on pair; do not make only the candidate's workload easier. Add output-quality checks before a production recommendation.

## Results and observability

### Artifacts and timing scope

Each real run writes under `benchmarks/<run_id>/`:

| File | Purpose |
| --- | --- |
| `metadata.json` | Status, endpoint, configuration evidence, local GPU topology, workload/drain windows |
| `requests.jsonl` | Per-request timings, usage, errors, scheduling, chunk counts, routing identity when available |
| `summary.csv` | Per-workload aggregates, including detailed router maps |
| `summary.md` | Compact human-readable table |
| `gpu_metrics.csv` | Local GPU samples, when available |
| `vllm_metrics.jsonl` | Periodic and workload-boundary server/router metrics |
| `gpu_metrics_unavailable.txt` | Explanation when local GPU sampling cannot start |

Client latency and TTFT start immediately before HTTP send, after payload construction. `scheduled_latency_s` includes delay from the planned open-loop arrival. TPOT is `(latency - TTFT) / (output_tokens - 1)` when defined, not directly measured GPU decode time.

The field called inter-token latency measures gaps between **SSE content chunks**. A chunk can contain several tokens, especially with speculation. Inspect `observed_stream_chunks` and server-reported usage; chunk intervals are not exact per-token timestamps.

If TTFT is almost total latency while TPOT/chunk intervals round to zero, suspect buffering and compare direct/routed streams. The main router flushes SSE lines, but another proxy can still buffer. Earlier buffered runs cannot have valid TTFT reconstructed from their aggregate summaries; rerun them.

GPU memory max is the maximum observed device sample, not a deployment-wide sum. Allocation can include reserved KV-cache space. Imbalance needs at least two observed GPUs in the relevant scope; missing remote data is not perfect balance. Scrape failures, missing boundaries, or counter resets also make zero deltas insufficient proof of “no errors.”

### Summaries and reports

```bash
python3 scripts/summarize_results.py benchmarks/baseline_r1

python3 scripts/analyze_month5.py \
  benchmarks/m5_single_r1 \
  benchmarks/m5_tp2_r1 \
  benchmarks/m5_replicas_round_robin_r1 \
  --out reports/report-05-topology-results.md

python3 scripts/analyze_month5.py \
  benchmarks/m5_spec_off_r1 \
  benchmarks/m5_spec_ngram4_r1 \
  --out reports/report-05-speculative-results.md
```

The Month 5 analyzer uses the first row for each matching workload as baseline: supply the control first. It does not compute repetition medians, enforce every experimental control, or merge remote GPU files; review these separately before completing [Report 05](reports/report-05-parallelism-decoding-routing.md).

Other entry points are `scripts/analyze_saturation.py`, `scripts/analyze_month4.py`, and `scripts/diagnose_incident.py`; use `--help` and the corresponding plan. Reports should include exact environment, workload/configuration evidence, results, failure analysis, limitations, and the workload/SLO boundary of a recommendation.

### Optional live dashboards

Local benchmark artifacts do not require Prometheus/Grafana. An optional stack is defined in [observability/docker-compose.yml](observability/docker-compose.yml):

```bash
docker compose -f observability/docker-compose.yml up -d
```

Use a Docker-capable NVIDIA host with GPU-container support; this is not a prerequisite inside Runpod. The supplied stack targets vLLM on host port `8000`, an optional benchmark exporter on `9001`, and DCGM for GPU telemetry. Edit scrape targets for remote workers or router port `9000`; Pods are not automatically discovered.

Add `--metrics-export-port 9001` to a benchmark for live client metrics. Prometheus uses `9090`, Grafana `3000`, and DCGM exporter `9400`. Compose currently uses `latest` images and default Grafana `admin`/`admin` credentials: pin images, change credentials, and restrict access before shared use. See [dashboard notes](dashboards/month3-observability.md).

## Troubleshooting

| Symptom | Check and action |
| --- | --- |
| Expected TP=2, observed TP=1 | Use TP=1 expectations for the control, or restart a two-GPU server with explicit TP=2. Labels never reconfigure it. |
| Drain timeout with connection refused | Check URL, Pod identity, port, and process lifetime. Query `/health` and `/metrics` from the benchmark shell. Do not bypass the guard. |
| Router works on A, localhost:9000 fails on B | Loopback means the current Pod. Run the benchmark on A or deliberately configure a reachable router bind/address. |
| Private DNS fails | Check both Pods' networking support/settings and exact IDs. Public DNS working is a separate check. |
| Public URL works with curl, router gets 403 | Compare raw status/body, authentication, and proxy behavior from the same caller. Use the verified private path for controlled measurements. |
| `Expecting value: line 1 column 1` | The JSON formatter got empty/non-JSON data. Inspect with `curl -i` without the pipe. SSE is not one JSON document. |
| TTFT nearly equals latency; TPOT near zero | Check incremental delivery through every intermediary and the fault-proxy limitation. Rerun affected measurements. |
| Two replicas, `gpu_count_observed=1` | Local telemetry sees one host. Require per-worker counters and GPU samples from both serving Pods. |
| Router healthy with zero attempts | Send traffic and inspect worker identities/counters; readiness has not been proven for both workers. |
| Router queue gauges absent | A worker did not supply valid running/waiting metrics. Inspect each `/metrics`; unknown is not idle. |
| Different average output lengths | EOS/stopping changed the amount of work. Use paired output policies and verify actual usage. |
| Port occupied or unexpected GPU memory use | Inspect existing processes; an auto-started server may own the port/GPU. |
| Run ID already exists | Preserve prior evidence and choose a new ID; the harness refuses to overwrite it. |

## Local validation and project layout

From the repository root, without a GPU server:

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s pytorch_baseline -p 'test_*.py' -v

for workload in workloads/*.json; do
  python3 scripts/benchmark_vllm.py --workload "$workload" --dry-run || break
done
```

Router integration tests bind localhost ports; allow local networking in restricted environments. Passing tests validates harness/router behavior, not H100 performance or Runpod connectivity.

```text
scripts/             Benchmark client, collectors, analyzers, incident diagnosis
workloads/           Version-controlled experiment definitions
routing/             Replica router and experimental fault proxy
pytorch_baseline/    Generation, profiling, precision, and comparison tools
observability/       Prometheus, Grafana, DCGM configuration
dashboards/          Metric and dashboard notes
tests/               Harness, router, and analysis tests
notes/               Month-by-month experimental plans
incidents/           Incident investigation templates
reports/             Report templates and curated analyses
benchmarks/          Generated run artifacts (ignored by Git)
```

Back up raw runs, workload copies, server/router logs, worker inventories, and environment evidence before releasing rented GPUs. `benchmarks/`, `.env`, and logs are ignored by Git, so a commit is not a results backup. Review artifacts for credentials, endpoint details, and sensitive prompt/error content before sharing.
