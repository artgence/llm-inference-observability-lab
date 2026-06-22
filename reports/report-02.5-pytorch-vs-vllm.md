# Report 02.5: PyTorch Baseline vs vLLM Serving - Latency, Memory, and Profiling

## Summary

Status: awaiting GPU-backed PyTorch and matching vLLM runs.

This report compares plain Hugging Face/PyTorch `generate()` with vLLM to explain the operational value of continuous batching, managed KV cache, streaming, scheduling, and serving-oriented observability. The objective is understanding, not proving that one implementation always wins.

## Environment and Fairness Controls

| Field | PyTorch | vLLM |
| --- | --- | --- |
| Model repository/revision | TBD | TBD |
| Tokenizer/chat template | TBD | TBD |
| Precision/quantization | TBD | TBD |
| GPU | TBD | TBD |
| PyTorch/Transformers/vLLM version | TBD | TBD |
| Prompt/output policy | TBD | TBD |
| Warmups/measured repetitions | TBD | TBD |
| Timing scope | in-process | client E2E including HTTP/queueing |

Reject direct comparisons when the model revision, rendered prompt/token count, dtype, or output-token behavior differs materially.

## Generation Results

| Runtime | Batch/concurrency | Prompt tokens | Output tokens | Latency p50 | Latency p95 | Output tokens/sec | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vLLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## CUDA Memory Results

| Runtime/case | Model weights | Allocated after load | Peak allocated | Peak reserved | Estimated KV cache | Other/activation estimate | Device/process memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| vLLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

PyTorch allocator values do not include all memory allocated directly by CUDA libraries or NCCL. State the source and visibility of each memory number.

## Batch-Size Sensitivity

- Latency change from batch 1 to 8: TBD
- Throughput change from batch 1 to 8: TBD
- Peak-memory change: TBD
- First unacceptable/OOM case: TBD
- Why vLLM continuous batching behaves differently from one static PyTorch batch: TBD

## Prompt-Length Sensitivity

- 512-token result: TBD
- 2,048-token result: TBD
- 8,192-token result: TBD
- Prefill-time growth: TBD
- KV/activation-memory growth: TBD

## FP16 vs BF16

| Dtype | Supported | Latency | Tokens/sec | Peak allocated | Output sanity |
| --- | --- | ---: | ---: | ---: | --- |
| FP16 | TBD | TBD | TBD | TBD | TBD |
| BF16 | TBD | TBD | TBD | TBD | TBD |

Do not report silent fallback as a valid dtype result.

## Prefill and Decode

- Prefill timing and memory: TBD
- Decode timing and memory: TBD
- KV-cache bytes after prefill/decode: TBD
- What remains allocated between tokens: TBD
- What vLLM changes through paged KV management and scheduling: TBD

## Profiler Findings

- Representative case: TBD
- Dominant CPU operators: TBD
- Dominant CUDA operators/kernels: TBD
- Attention/operator shapes: TBD
- Trace path: TBD
- Profiler overhead caveat: TBD

## Why a Serving Engine Exists

- Static batching limitation observed: TBD
- Naive KV-cache allocation limitation observed: TBD
- Queueing/scheduling capability missing from in-process `generate()`: TBD
- vLLM benefit supported by measurements: TBD
- vLLM overhead supported by measurements: TBD

## Conclusion

- Which operators dominated inference: TBD
- How batch size changed memory: TBD
- How prompt length changed memory: TBD
- Weight versus activation versus KV-cache interpretation: TBD
- Operationally credible PyTorch skill demonstrated: TBD
