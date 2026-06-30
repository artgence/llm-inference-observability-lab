# Report 04: Practical vLLM Serving Knobs and Capacity Boundaries

## Summary

Status: awaiting Month 4 GPU-backed runs.

This report evaluates prefix caching, chunked prefill, continuous batching limits, context capacity, GPU-memory allocation, admission control, and an optional multi-GPU FP8 70B deployment.

## Environment

| Field | Value |
| --- | --- |
| Date | TBD |
| GPU and count | TBD |
| Model | TBD |
| vLLM version | TBD |
| CUDA/driver | TBD |
| Effective GPU hourly cost | TBD |
| Actual vLLM launch command | TBD from `metadata.json` |
| `/metrics` cache config | TBD from `vllm:cache_config_info` |
| Config expectations verified | TBD |

The server configuration label is descriptive only. Do not interpret a comparison
unless the actual command/config evidence was captured and all expected values matched.

## Prefix Caching

- Shared-prefix TTFT effect: TBD
- Shared-prefix throughput effect: TBD
- Prefix-cache hit/query tokens and hit rate: TBD
- Prompt-token parity delta and pass/fail: TBD
- Decode/TPOT effect: TBD
- Cache/KV tradeoff: TBD

## Chunked Prefill and Context Capacity

- A (prefix off, chunked off) versus B (prefix off, chunked on): TBD
- B (prefix off, chunked on) versus C (prefix on, chunked on): TBD
- Short-request TTFT during long prefills: TBD
- Long-prompt completion throughput: TBD
- Queue/KV boundary: TBD

## Continuous Batching and Memory

- Best tested `MAX_NUM_SEQS`: TBD
- Throughput plateau: TBD
- First unacceptable p99/queue point: TBD
- Selected `GPU_MEMORY_UTILIZATION`: TBD

## Admission Control

- No-policy impact: TBD
- Reject-policy impact: TBD
- Truncate-policy impact: TBD
- Recommended maximum prompt policy: TBD

## Quantized 70B

- Hardware topology: TBD
- FP8 memory footprint: TBD
- Throughput/latency/cost result: TBD
- Observable quality comparison and method: TBD

## Recommended Serving Configuration

```bash
TBD
```

## Conclusion

- Knob with the largest measured benefit: TBD
- Capacity boundary: TBD
- Cost/latency tradeoff: TBD
- Production guardrail: TBD
- Drain guards and before/after counter windows valid: TBD
