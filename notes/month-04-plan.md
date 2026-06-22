# Month 4 Plan: PyTorch Inference Foundations and vLLM Serving Knobs

Goal: understand the Hugging Face/PyTorch inference path and CUDA memory behavior, then measure how vLLM scheduling, cache, memory, and admission knobs change TTFT, TPOT, throughput, failures, and cost.

## Implementation Status

- [x] Add `pytorch_baseline/hf_generate.py` with deterministic Hugging Face `generate()` timing and token-throughput output.
- [x] Add `pytorch_baseline/memory_profile.py` with allocated/reserved/peak CUDA memory and prefill/decode phase markers.
- [x] Add `pytorch_baseline/attention_shapes.py` for bounded tensor-shape inspection without retaining full activations.
- [x] Add `pytorch_baseline/precision_compare.py` for supported FP16/BF16 comparisons.
- [x] Add `pytorch_baseline/compare_vllm.py` to join PyTorch JSON with vLLM summary rows.
- [x] Add `pytorch_baseline/README.md` with reproducible commands, methodology, and limitations.
- [x] Add PyTorch Profiler CPU/CUDA traces, operator tables, shape recording, and tensor-memory reporting.
- [x] Add the matching vLLM p512/p2048/p8192 and c1/c2/c4/c8 workload matrix.
- [x] Add the Report 02.5 template and comparison-table generator.
- [ ] Complete Report 02.5 with a same-model PyTorch-versus-vLLM comparison.
- [x] Add explicit launcher controls for prefix caching, chunked prefill, batching limits, KV-cache dtype, GPU-memory utilization, CPU offload, and quantization.
- [x] Add repeated-versus-unique prefix generation with equal prompt lengths.
- [x] Add continuous-batching, context-capacity, admission-control, and quantized-70B workloads.
- [x] Add workload-level mixed prompt lengths and reject/truncate admission policies.
- [x] Add server-configuration labels and a Month 4 comparison report generator.
- [ ] Define a small task-quality evaluation set before claiming an 8B/70B quality tradeoff.
- [ ] Execute each server variant on the GPU host.
- [ ] Complete Report 04 with measured boundaries and selected production defaults.

Do not launch a second vLLM process while the existing PID 1 server owns port 8000. Restart or recreate the serving container between server-side variants. Use the same model, workload, GPU cost, and benchmark options for both sides of each comparison.

## Foundation Track: Hugging Face/PyTorch Baseline

Build this small module before interpreting the vLLM knob results:

```text
pytorch_baseline/
  hf_generate.py
  memory_profile.py
  attention_shapes.py
  precision_compare.py
  compare_vllm.py
  README.md
```

The goal is not to beat vLLM. The goal is to understand the model-loading, tokenization, forward-pass, attention, CUDA-memory, prefill, and decode behavior that a serving engine must manage.

### Model and Fairness Rules

- Prefer the exact model repository and revision used by the vLLM baseline.
- If the current compressed/FP8 checkpoint is not supported by plain Transformers, select a smaller Transformers-compatible checkpoint and rerun that exact revision in both PyTorch and vLLM.
- Keep tokenizer revision, chat template, prompt token IDs, output cap, greedy/deterministic decoding, dtype, and device fixed.
- Report that PyTorch timing is in-process while the vLLM benchmark includes HTTP/queueing. Show compute-oriented and client end-to-end measurements separately rather than treating them as identical scopes.
- Use `model.eval()` and `torch.inference_mode()`. Synchronize CUDA before and after timed regions.
- Warm up each shape, reset peak-memory statistics, then run at least three measured repetitions.

### Baseline Matrix

| Dimension | Initial values |
| --- | --- |
| Batch size | 1, 2, 4, 8 |
| Prompt target | 512, 2,048, 8,192 tokens |
| Output target | 128 tokens |
| Precision | FP16, BF16 where supported |
| Cache | Hugging Face generation cache enabled; document implementation/default |

Start with batch 1 and p512/o128. Increase one dimension at a time and stop before OOM rather than silently changing another variable.

### Script Responsibilities

`hf_generate.py`:

- load tokenizer/model with explicit revision and dtype
- apply the model chat template
- run deterministic `generate(max_new_tokens=...)`
- report input/output tokens, wall time, latency, and output tokens/sec as JSON/CSV

`memory_profile.py`:

- record memory before model load, after model load, after prompt transfer, after prefill, and after decode
- report `memory_allocated`, `memory_reserved`, `max_memory_allocated`, and `max_memory_reserved`
- complement PyTorch allocator metrics with per-process/device memory where available
- state that PyTorch allocator snapshots do not include every direct CUDA/NCCL allocation

`attention_shapes.py`:

- record bounded module/input/output shape metadata for representative attention layers
- avoid storing full tensors or registering hooks across every layer for long runs
- connect batch, sequence, head, and head-dimension shapes to activation/KV growth

`precision_compare.py`:

- run the same shapes under FP16 and BF16
- capture startup success, latency, throughput, allocated/peak memory, and output sanity
- report unsupported kernels/dtypes as explicit results rather than automatically falling back

### PyTorch Profiler

Profile representative small and medium cases rather than every benchmark iteration. Enable CPU and CUDA activities, `record_shapes`, and `profile_memory`; export a Chrome/Perfetto trace and a sorted operator table.

Use phase annotations for:

- model/tokenizer setup outside measured inference
- tokenization and host-to-device transfer
- prefill/first forward pass
- decode/token loop

Practice answering from evidence:

- Which operators and CUDA kernels dominate inference?
- How does batch size change latency and peak memory?
- How does prompt length change activation and KV-cache memory?
- What is allocated during prefill versus decode?
- What memory belongs to model weights, activations, KV cache, allocator reserve, and non-PyTorch CUDA users?
- Why do continuous batching, paged KV-cache management, and prefix caching exist in vLLM?

### PyTorch/vLLM Reading Map

Read only enough code and documentation to connect observations to:

- model loading and Hugging Face compatibility
- tensor dtype and CUDA allocation
- tokenizer/chat-template path
- attention backend selection
- KV-cache allocation and reuse
- `torch.compile` and CUDA graph basics
- vLLM batching, metrics, parameter sweeps, quantization, prefix caching, and supported models

### Report 02.5

Deliver `Report 02.5: PyTorch Baseline vs vLLM Serving - Latency, Memory, and Profiling` with:

- same-model/config comparison table
- batch and prompt sensitivity results
- FP16/BF16 findings
- allocated/reserved/peak memory table
- profiler operator summary and one representative trace
- prefill/decode memory explanation
- clear explanation of why the naive baseline differs from a serving engine

Do not spend this track training CNNs, implementing backpropagation, or studying JAX deeply. Do not claim “PyTorch expert.” Credible scope descriptions are “PyTorch inference/profiling,” “PyTorch CUDA memory analysis,” “Hugging Face/PyTorch serving baselines,” and “PyTorch Profiler for inference performance analysis.”

## Experiment 1: Prefix Caching

The workload compares a request-unique 1,536-token prefix with a shared 1,536-token prefix inside equal 2,048-token prompts. Run it once with automatic prefix caching explicitly disabled and once enabled.

```bash
ENABLE_PREFIX_CACHING=false scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_prefix_cache.json \
  --server-config-label prefix_cache_off

ENABLE_PREFIX_CACHING=true scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_prefix_cache.json \
  --server-config-label prefix_cache_on
```

Compare TTFT and `vllm_prefix_cache_hit_rate` first. Automatic prefix caching reuses prefill KV blocks; it should not be presented as a decode/TPOT optimization.

## Experiment 2: Chunked Prefill and Long Context

Run the same 2K, 8K, and 15,360-token prompt sweep with chunked prefill disabled and enabled. Keep `MAX_NUM_BATCHED_TOKENS` fixed so the comparison is attributable.

```bash
ENABLE_CHUNKED_PREFILL=false MAX_NUM_BATCHED_TOKENS=8192 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_context_capacity.json \
  --server-config-label chunked_prefill_off

ENABLE_CHUNKED_PREFILL=true \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_NUM_PARTIAL_PREFILLS=2 \
MAX_LONG_PARTIAL_PREFILLS=1 \
LONG_PREFILL_TOKEN_THRESHOLD=4096 \
scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_context_capacity.json \
  --server-config-label chunked_prefill_on
```

## Experiment 3: Continuous Batching and Capacity

Run the same concurrency sweep with explicit sequence limits. Compare where throughput flattens and TTFT, queue depth, or KV-cache pressure accelerates.

```bash
MAX_NUM_SEQS=32 GPU_MEMORY_UTILIZATION=0.90 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_batching_capacity.json \
  --server-config-label max_seqs_32

MAX_NUM_SEQS=64 GPU_MEMORY_UTILIZATION=0.90 scripts/start_vllm_server.sh
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_batching_capacity.json \
  --server-config-label max_seqs_64
```

## Experiment 4: Admission Control

This client-side simulation repeats a mixed prompt pattern of 512, 512, 2,048, and 15,360 tokens. It compares no policy, rejecting prompts over 8,192 tokens, and truncating them to 8,192 tokens.

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_admission_control.json \
  --server-config-label admission_control
```

`admission_rejected_count` is an intentional policy result. Do not combine it with server OOM, timeout, or 5xx failures when judging reliability.

## Experiment 5: Quantized 70B

The FP8 70B checkpoint is approximately half the BF16 weight footprint but still does not fit in one 46 GB L40S. Use at least two L40S GPUs and tensor parallelism; skip this experiment when only one GPU is available.

```bash
MODEL_ID=neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=16384 \
scripts/start_vllm_server.sh

MODEL_ID=neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 \
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_quantized_70b.json \
  --server-config-label llama31_70b_fp8_tp2
```

## Build Report 04

Supply baseline variants before candidate variants so percentage comparisons use the intended reference.

```bash
python3 scripts/analyze_month4.py \
  benchmarks/PREFIX_OFF_RUN \
  benchmarks/PREFIX_ON_RUN \
  benchmarks/CHUNKED_OFF_RUN \
  benchmarks/CHUNKED_ON_RUN \
  benchmarks/MAX_SEQS_32_RUN \
  benchmarks/MAX_SEQS_64_RUN \
  benchmarks/ADMISSION_RUN \
  --out reports/report-04-results.md
```

## Acceptance Checks

- PyTorch and vLLM comparisons use the same model revision, token IDs/chat template, prompt/output shapes, dtype, and deterministic generation settings.
- PyTorch timing uses CUDA synchronization, warmups, peak-memory resets, and at least three measured repetitions.
- Profiler overhead is excluded from ordinary latency/throughput results and reported separately.
- Memory conclusions distinguish PyTorch allocated/reserved memory from total device/process memory.
- Report 02.5 explains prefill/decode behavior and why serving-engine KV/batching controls exist without claiming kernel-level expertise.
- All server variants record a distinct `server_config_label` in `metadata.json`.
- Prefix prompts have the same total target and differ only in prefix reuse.
- Prefix-cache claims include non-zero cache query/hit evidence from vLLM metrics.
- Scheduler-delay p95 remains low enough that the client is not the bottleneck.
- Context targets plus output caps remain below `MAX_MODEL_LEN=16384`.
- Any claimed improvement includes TTFT/TPOT, throughput, errors, waiting requests, KV usage, GPU memory, and cost evidence.
- The chosen serving defaults state the tested GPU/model boundary and do not generalize a two-GPU 70B result to one L40S.

## References

- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch CUDA memory analysis](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)
- [Hugging Face generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args.html)
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- [vLLM quantization support](https://docs.vllm.ai/en/stable/features/quantization/index.html)
