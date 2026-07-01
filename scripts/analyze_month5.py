#!/usr/bin/env python3
"""Build the Month 5 sharding, replication, decoding, and routing report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DISPLAY_COLUMNS = [
    "deployment_label",
    "deployment_type",
    "deployment_gpu_count",
    "tensor_parallel_size",
    "workload",
    "latency_p99_s",
    "ttft_p95_s",
    "tpot_p95_s",
    "requests_per_sec",
    "output_tokens_per_sec",
    "throughput_speedup",
    "scaling_efficiency",
    "gpu_memory_used_mb_max",
    "gpu_memory_used_imbalance_mb",
    "gpu_utilization_imbalance_pct",
    "vllm_spec_decode_acceptance_rate",
    "router_retries",
    "router_failures",
    "router_worker_attempt_imbalance_pct",
    "router_circuit_open_workers_max",
    "error_rate",
    "cost_per_1m_total_tokens_usd",
    "cost_per_successful_request_usd",
]


def safe_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a" if value in (None, "") else str(value)
    return f"{number:.3f}"


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"
    if not summary_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"{run_dir} must contain summary.csv and metadata.json"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = list(csv.DictReader(handle))
    deployment = metadata.get("deployment") or {}
    parallelism = metadata.get("parallelism") or {}
    evidence = metadata.get("server_evidence") or {}
    comparisons = evidence.get("config_comparisons") or []
    verified = bool(comparisons) and all(
        comparison.get("matched") for comparison in comparisons
    )
    label = metadata.get("server_config_label") or run_dir.name
    gpu_count = deployment.get("gpu_count")
    if gpu_count is None:
        gpu_count = (metadata.get("gpu_topology") or {}).get("gpu_count")
    for row in rows:
        row.update(
            {
                "deployment_label": label,
                "deployment_type": deployment.get("type") or "unspecified",
                "deployment_gpu_count": gpu_count,
                "tensor_parallel_size": parallelism.get(
                    "tensor_parallel_size", 1
                ),
                "pipeline_parallel_size": parallelism.get(
                    "pipeline_parallel_size", 1
                ),
                "server_config_verified": verified,
                "server_launch_command": (
                    (evidence.get("launch") or {}).get("command")
                    or "unavailable"
                ),
                "topology_matrix": (
                    metadata.get("gpu_topology") or {}
                ).get("topology_matrix"),
                "run_id": metadata.get("run_id", run_dir.name),
            }
        )
    return rows


def add_scaling_metrics(rows: list[dict[str, Any]]) -> None:
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row.get("workload"))].append(row)
    for workload_rows in by_workload.values():
        baseline = workload_rows[0]
        baseline_throughput = safe_float(
            baseline.get("output_tokens_per_sec")
        )
        baseline_gpus = safe_float(baseline.get("deployment_gpu_count"))
        for row in workload_rows:
            throughput = safe_float(row.get("output_tokens_per_sec"))
            gpu_count = safe_float(row.get("deployment_gpu_count"))
            speedup = (
                throughput / baseline_throughput
                if throughput is not None
                and baseline_throughput not in (None, 0)
                else None
            )
            gpu_ratio = (
                gpu_count / baseline_gpus
                if gpu_count is not None and baseline_gpus not in (None, 0)
                else None
            )
            row["throughput_speedup"] = speedup
            row["scaling_efficiency"] = (
                speedup / gpu_ratio
                if speedup is not None and gpu_ratio not in (None, 0)
                else None
            )


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        column
        for column in DISPLAY_COLUMNS
        if any(row.get(column) not in (None, "") for row in rows)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(fmt(row.get(column)) for column in columns)
            + " |"
        )
    return lines


def build_findings(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_workload[str(row.get("workload"))].append(row)
    for workload, workload_rows in sorted(by_workload.items()):
        if len(workload_rows) < 2:
            continue
        baseline = workload_rows[0]
        for candidate in workload_rows[1:]:
            if not (
                baseline.get("server_config_verified")
                and candidate.get("server_config_verified")
            ):
                findings.append(
                    f"- **{workload}: comparison rejected:** command/config "
                    "expectations were not verified for both deployments."
                )
                continue
            findings.append(
                f"- **{workload}: {candidate['deployment_label']} versus "
                f"{baseline['deployment_label']}:** output-throughput speedup "
                f"{fmt(candidate.get('throughput_speedup'))}x; scaling efficiency "
                f"{fmt(candidate.get('scaling_efficiency'))}; TTFT p95 "
                f"{fmt(candidate.get('ttft_p95_s'))}s versus "
                f"{fmt(baseline.get('ttft_p95_s'))}s; cost per 1M total tokens "
                f"{fmt(candidate.get('cost_per_1m_total_tokens_usd'))} versus "
                f"{fmt(baseline.get('cost_per_1m_total_tokens_usd'))}."
            )
    router_rows = [
        row for row in rows if row.get("deployment_type") == "replicas"
    ]
    if router_rows:
        worst = max(
            router_rows,
            key=lambda row: safe_float(row.get("latency_p99_s")) or 0,
        )
        findings.append(
            f"- **Replica routing:** worst observed p99 was "
            f"{fmt(worst.get('latency_p99_s'))}s; retries "
            f"{fmt(worst.get('router_retries'))}, router failures "
            f"{fmt(worst.get('router_failures'))}, and worker-attempt imbalance "
            f"{fmt(worst.get('router_worker_attempt_imbalance_pct'))}%; cost per "
            f"successful request "
            f"{fmt(worst.get('cost_per_successful_request_usd'))}."
        )
    speculative_rows = [
        row
        for row in rows
        if safe_float(row.get("vllm_spec_decode_draft_tokens")) not in (None, 0)
        and row.get("server_config_verified")
    ]
    for row in speculative_rows:
        findings.append(
            f"- **Speculative decoding ({row['deployment_label']}, "
            f"{row['workload']}):** acceptance rate "
            f"{fmt(row.get('vllm_spec_decode_acceptance_rate'))}, TPOT p95 "
            f"{fmt(row.get('tpot_p95_s'))}s, output throughput "
            f"{fmt(row.get('output_tokens_per_sec'))} tokens/s."
        )
    if not findings:
        findings.append(
            "- Add at least two controlled deployment runs with shared workload names."
        )
    return findings


def render_evidence(rows: list[dict[str, Any]]) -> list[str]:
    deployments: dict[str, dict[str, Any]] = {}
    for row in rows:
        deployments.setdefault(
            str(row["deployment_label"]),
            {
                "type": row.get("deployment_type"),
                "gpus": row.get("deployment_gpu_count"),
                "tp": row.get("tensor_parallel_size"),
                "verified": row.get("server_config_verified"),
                "command": row.get("server_launch_command"),
                "topology": row.get("topology_matrix"),
            },
        )
    lines = [
        "| deployment | type | GPUs | TP | config verified | command | topology captured |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for label, evidence in deployments.items():
        command = str(evidence["command"]).replace("|", "\\|")
        lines.append(
            f"| {label} | {evidence['type']} | {fmt(evidence['gpus'])} | "
            f"{fmt(evidence['tp'])} | {evidence['verified']} | `{command}` | "
            f"{bool(evidence['topology'])} |"
        )
    return lines


def build_report(rows: list[dict[str, Any]]) -> str:
    add_scaling_metrics(rows)
    return "\n".join(
        [
            "# Report 05: Parallelism, Decoding, and Routing Trade-offs in vLLM Serving",
            "",
            "## Findings",
            "",
            *build_findings(rows),
            "",
            "## Deployment Evidence",
            "",
            *render_evidence(rows),
            "",
            "## Results",
            "",
            *render_table(rows),
            "",
            "## Decision",
            "",
            "- Use TP when model-fit or KV-cache capacity requires sharding and measured latency/cost remains acceptable.",
            "- Prefer independent replicas when the model fits per GPU and throughput, isolation, and failover dominate.",
            "- Treat PP as a secondary fit/topology option and EP as conceptual unless an MoE role requires it.",
            "- Do not recommend speculative decoding without measured acceptance, TPOT, throughput, and memory evidence.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Run directories in comparison order, single-GPU baseline first.",
    )
    parser.add_argument(
        "--out",
        default="reports/report-05-results.md",
    )
    args = parser.parse_args()
    rows = [
        row for run_dir in args.run_dirs for row in load_run(Path(run_dir))
    ]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
