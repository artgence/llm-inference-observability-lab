# Hugging Face/PyTorch Inference Baseline

This module provides an in-process inference baseline for understanding why a serving engine such as vLLM exists. It is not intended to outperform vLLM or establish training expertise.

## Environment

Use the same GPU host and, when supported, the exact model revision used by the vLLM comparison.

First verify that the PyTorch already installed in the vLLM environment can use the
GPU. An unpinned `pip install torch` can replace it with a build that requires a
newer NVIDIA driver.

```bash
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python3 -m pip install -r pytorch_baseline/requirements.txt
export HF_TOKEN=hf_your_token_here
export PYTORCH_MODEL_ID=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8
export MODEL_REVISION=main
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_XET_HIGH_PERFORMANCE=1
```

The CUDA check must end with `True`. If it is `False` and PyTorch reports that the
driver is too old, either update the host driver or restore the CUDA-compatible
PyTorch version required by the vLLM image. Do not use CPU fallback for the FP8
checkpoint: compressed execution may be disabled and missing quantization state
may be initialized, making correctness and performance results invalid.

The current vLLM checkpoint may require a specialized quantization integration when loaded through plain Transformers. If it cannot run, select one smaller Transformers-compatible checkpoint and use that exact repository/revision for both the PyTorch and vLLM comparison. Do not compare different models and attribute the result to the serving engine.

## Validate Without Loading a Model

```bash
python3 pytorch_baseline/hf_generate.py --dry-run
python3 pytorch_baseline/memory_profile.py --dry-run
python3 pytorch_baseline/attention_shapes.py --dry-run
python3 pytorch_baseline/precision_compare.py --dry-run
```

## Generation Matrix

Run deterministic Hugging Face `generate()` across the initial batch/prompt matrix:

```bash
python3 pytorch_baseline/hf_generate.py \
  --dtype bf16 \
  --batch-sizes 1,2,4,8 \
  --prompt-tokens 512,2048,8192 \
  --max-new-tokens 128 \
  --warmups 1 \
  --repetitions 3 \
  --out benchmarks/pytorch/hf_generate_bf16.json
```

The default forces 128 output tokens to make throughput cases comparable. Use `--no-force-output-length` when matching a vLLM run that allows EOS before the cap, and compare actual output-token counts.

The result JSON includes a `prompt_manifest` with synthetic user content, rendered chat token IDs, and SHA-256 fingerprints. Use it to reproduce or audit the exact PyTorch prompt instead of relying only on nominal prompt length.

## Prefill, Decode, Memory, and Profiler

Run ordinary phase/memory measurement without profiler overhead:

```bash
python3 pytorch_baseline/memory_profile.py \
  --dtype bf16 \
  --batch-size 1 \
  --prompt-tokens 2048 \
  --new-tokens 128 \
  --out benchmarks/pytorch/memory_p2048.json
```

Run a separate representative profiler case:

```bash
python3 pytorch_baseline/memory_profile.py \
  --dtype bf16 \
  --batch-size 1 \
  --prompt-tokens 2048 \
  --new-tokens 128 \
  --profile \
  --trace-out benchmarks/pytorch/inference_trace.json \
  --operator-table-out benchmarks/pytorch/operator_table.txt \
  --memory-snapshot benchmarks/pytorch/cuda_memory_snapshot.pickle \
  --out benchmarks/pytorch/profile_p2048.json
```

Profiler results are diagnostic and must not be mixed into ordinary latency numbers. CUDA memory snapshots only see allocations managed by the PyTorch allocator; compare them with process/device telemetry for direct CUDA or NCCL allocations.

## Attention Shapes

```bash
python3 pytorch_baseline/attention_shapes.py \
  --device cuda \
  --dtype bf16 \
  --batch-size 1 \
  --prompt-tokens 512 \
  --max-modules 2 \
  --out benchmarks/pytorch/attention_shapes.json
```

Only shape, dtype, and device metadata is retained. Full activation tensors are not stored.

## FP16 vs BF16

Each precision runs in a fresh process so allocator/model state does not leak between comparisons:

```bash
python3 pytorch_baseline/precision_compare.py \
  --dtypes fp16,bf16 \
  --batch-sizes 1,4 \
  --prompt-tokens 512,2048 \
  --max-new-tokens 128 \
  --out benchmarks/pytorch/precision_compare.json
```

Unsupported dtypes/checkpoints are recorded as failed cases; the script does not silently fall back.

## vLLM Comparison

Run the matching vLLM shape/concurrency workload:

```bash
python3 scripts/benchmark_vllm.py \
  --workload workloads/month4_pytorch_vllm_comparison.json \
  --server-config-label report02_5_vllm
```

Join the completed results:

```bash
python3 pytorch_baseline/compare_vllm.py \
  benchmarks/pytorch/hf_generate_bf16.json \
  benchmarks/VLLM_RUN_ID \
  --out reports/report-02.5-comparison-results.md
```

Static PyTorch batch size and vLLM concurrent requests are not identical scheduling mechanisms. Report both scopes explicitly:

- PyTorch: in-process model execution and generation latency.
- vLLM: client end-to-end latency including HTTP, queueing, continuous batching, and streaming.

For strict prompt equivalence, preserve prompt text/token fingerprints from the PyTorch output and verify the vLLM server-reported prompt-token count. Reject a comparison when model revision, tokenizer/chat template, dtype, output policy, or actual prompt/output lengths materially differ.

## Deliverable

Complete `reports/report-02.5-pytorch-vs-vllm.md` with:

- latency and tokens/sec
- allocated/reserved/peak CUDA memory
- FP16/BF16 results
- batch and prompt sensitivity
- representative operator table/trace
- prefill/decode and weights/activations/KV-cache explanation
- explicit timing-scope and memory-visibility caveats

Credible project wording is “PyTorch inference/profiling,” “PyTorch CUDA memory analysis,” or “Hugging Face/PyTorch serving baseline,” not “PyTorch expert.”
