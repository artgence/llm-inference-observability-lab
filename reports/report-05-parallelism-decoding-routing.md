# Report 05: Parallelism, Decoding, and Routing Trade-offs in vLLM Serving

Status: awaiting Month 5 GPU-backed runs.

## Decision Question

For the tested model and workload, should the service use one larger GPU, tensor
parallelism across smaller GPUs, or independent replicas behind a router?

## Deployment Evidence

| Deployment | Exact command | GPUs | Interconnect | TP/PP/replicas | Config verified |
| --- | --- | ---: | --- | --- | --- |
| Single-GPU baseline | TBD | TBD | TBD | TBD | TBD |
| Tensor-parallel candidate | TBD | TBD | TBD | TBD | TBD |
| Replica candidate | TBD | TBD | TBD | TBD | TBD |

Record the topology matrix, CUDA/NCCL versions and settings, exact model revision,
aggregate hourly GPU cost, and vLLM `/metrics` configuration.

## Sharding Versus Replication

| Workload | Deployment | TTFT p95 | TPOT p95 | p99 | Output tok/s | GPU imbalance | Cost/1M tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- Model-fit or KV-cache reason for TP: TBD
- Observed TP/NCCL overhead: TBD
- Replica throughput and isolation result: TBD
- Selected architecture and rejected alternatives: TBD

## Routing and Failure Behavior

| Policy | Scenario | Achieved RPS | p99 | Timeouts | Retries | Worker imbalance | Circuit opens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round robin | Healthy | TBD | TBD | TBD | TBD | TBD | TBD |
| Least inflight | Healthy | TBD | TBD | TBD | TBD | TBD | TBD |
| Latency aware | Slow/failing replica | TBD | TBD | TBD | TBD | TBD | TBD |

State whether retry amplification increased load. Retries must remain bounded and
must occur before a response is sent.

## Speculative Decoding

| Workload | Baseline/speculative | Acceptance | TPOT p95 | p99 | Output tok/s | Memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Do not claim a benefit when acceptance is missing or when latency improves by
sacrificing saturated throughput/cost.

## NCCL and Topology Notes

- Collectives relevant to tested TP configuration: TBD
- PCIe/NVLink topology evidence: TBD
- NCCL/distributed errors observed: TBD
- Why TP was or was not worth its communication cost: TBD

## Final Recommendation

- Selected deployment: TBD
- Workload/SLO boundary: TBD
- Capacity and failover rationale: TBD
- Aggregate cost rationale: TBD
- Remaining uncertainty: TBD
