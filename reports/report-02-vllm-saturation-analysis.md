# Report 02: vLLM Operating-Point Robustness

## Summary

Status: awaiting Month 2 GPU-backed runs.

This report evaluates `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8` on one L40S across prompt length, output length, and burst shape at the Month 1 operating point of 18 RPS.

## Environment

| Field | Value |
| --- | --- |
| Date | TBD |
| GPU | NVIDIA L40S |
| GPU count | 1 |
| vLLM version | TBD |
| CUDA / driver | TBD |
| Model | neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 |
| Effective GPU hourly cost | TBD |
| Max model length | 8192 or actual |

## Controlled Sweeps

| Sweep | Values | Fixed variables |
| --- | --- | --- |
| Prompt target | 64, 128, 256, 512 | steady 18 RPS, output target 128 |
| Output target | 64, 128, 256, 512 | steady 18 RPS, prompt target 512 |
| Request pattern | steady, burst sizes 4, 8, 16 | average 18 RPS, prompt target 512, output target 128 |

`latency_s` and TTFT are measured from the actual HTTP send attempt. `scheduled_latency_s` is reported separately for open-loop analysis and includes any delay from the planned arrival time.

## Operating-Point Robustness

Paste or link the findings from `reports/report-02-results.md`.

```text
TBD
```

## Prefill Versus Decode

- Long prompt / short output result: TBD
- Short prompt / long output result: TBD
- TTFT versus TPOT evidence: TBD

## Queueing and Burst Behavior

- Steady-arrival result: TBD
- Burst-size sensitivity: TBD
- Load-generator scheduler validation: TBD

## GPU Memory and Utilization

- Peak memory utilization: TBD
- Average GPU utilization: TBD
- OOM or rejection boundary: TBD

## Throughput and Cost

- Best acceptable workload shape at 18 RPS: TBD
- Output-token throughput at that point: TBD
- Cost per million output tokens: TBD
- Cost assumptions and exclusions: TBD

## Conclusion

- Whether 18 RPS remains acceptable across Month 2 workload shapes: TBD
- Primary bottleneck: TBD
- Next experiment: TBD
