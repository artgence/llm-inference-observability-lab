#!/usr/bin/env python3
"""Analyze one or more benchmark runs and write a Month 2 operating-point report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DISPLAY_COLUMNS = [
    "workload",
    "concurrency",
    "target_request_rate_rps",
    "arrival_pattern",
    "burst_size",
    "prompt_tokens_avg",
    "output_tokens_avg",
    "output_tokens_target",
    "latency_p99_s",
    "ttft_p95_s",
    "tpot_p95_s",
    "requests_per_sec",
    "output_tokens_per_sec",
    "gpu_memory_utilization_pct_max",
    "error_rate",
    "cost_per_1m_output_tokens_usd",
]


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a" if value in (None, "") else str(value)
    return f"{number:.3f}"


def load_run(run_dir: Path) -> tuple[str, list[dict[str, str]]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata_path = run_dir / "metadata.json"
    label = run_dir.name
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        workload_file = metadata.get("workload_file")
        if workload_file:
            label = Path(workload_file).stem
    return label, rows


def varying_dimension(rows: list[dict[str, str]]) -> tuple[str, str]:
    candidates = [
        ("target_request_rate_rps", "Target RPS"),
        ("concurrency", "Concurrency"),
        ("prompt_tokens_target", "Prompt-token target"),
        ("max_tokens", "Output-token cap"),
        ("burst_size", "Burst size"),
    ]
    for field, label in candidates:
        values = {safe_float(row.get(field)) for row in rows}
        values.discard(None)
        if len(values) > 1:
            return field, label
    return "workload", "Workload"


def sorted_rows(rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    if field == "workload":
        return rows
    return sorted(rows, key=lambda row: safe_float(row.get(field)) or 0.0)


def detect_boundary(
    rows: list[dict[str, str]],
    dimension: str,
    latency_factor: float,
    scheduler_delay_threshold_s: float,
) -> tuple[dict[str, str] | None, list[str]]:
    if dimension == "workload" or not rows:
        return None, []
    baseline_p99 = safe_float(rows[0].get("latency_p99_s"))
    for index, row in enumerate(rows):
        reasons: list[str] = []
        latency_p99 = safe_float(row.get("latency_p99_s"))
        if (
            index > 0
            and baseline_p99
            and latency_p99
            and latency_p99 >= baseline_p99 * latency_factor
        ):
            reasons.append(f"p99 latency reached {latency_p99 / baseline_p99:.2f}x baseline")
        error_rate = safe_float(row.get("error_rate")) or 0.0
        if error_rate > 0:
            reasons.append(f"error rate reached {error_rate:.2%}")
        scheduler_delay = safe_float(row.get("scheduler_delay_p95_s")) or 0.0
        if scheduler_delay > scheduler_delay_threshold_s:
            reasons.append(
                f"load-generator scheduler delay reached {scheduler_delay:.3f}s p95"
            )
        if reasons:
            return row, reasons
    return None, []


def growth_ratio(first: dict[str, str], last: dict[str, str], field: str) -> float | None:
    start = safe_float(first.get(field))
    end = safe_float(last.get(field))
    if start is None or end is None or start <= 0:
        return None
    return end / start


def format_ratio(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def diagnose_bottleneck(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No rows available."
    first, last = rows[0], rows[-1]
    if (safe_float(last.get("oom_count")) or 0) > 0:
        return "GPU memory: OOM failures were observed at the high end of the sweep."
    memory_pct = safe_float(last.get("gpu_memory_utilization_pct_max"))
    if memory_pct is not None and memory_pct >= 95:
        return f"GPU memory/KV cache pressure: peak memory utilization reached {memory_pct:.1f}%."
    ttft_growth = growth_ratio(first, last, "ttft_p95_s")
    tpot_growth = growth_ratio(first, last, "tpot_p95_s")
    if ttft_growth is not None and ttft_growth >= 1.5 and (
        tpot_growth is None or ttft_growth >= tpot_growth * 1.2
    ):
        return (
            "Queueing or prefill pressure: TTFT p95 grew more sharply than TPOT p95 "
            f"({ttft_growth:.2f}x versus {format_ratio(tpot_growth)}x)."
        )
    if tpot_growth is not None and tpot_growth >= 1.5 and (
        ttft_growth is None or tpot_growth >= ttft_growth * 1.2
    ):
        return (
            "Decode pressure: TPOT p95 grew more sharply than TTFT p95 "
            f"({tpot_growth:.2f}x versus {format_ratio(ttft_growth)}x)."
        )
    return "Mixed or inconclusive from aggregate metrics; inspect request logs and vLLM metrics."


def render_table(rows: list[dict[str, str]]) -> list[str]:
    columns = [column for column in DISPLAY_COLUMNS if any(column in row for row in rows)]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return lines


def build_report(
    runs: list[tuple[str, list[dict[str, str]]]],
    latency_factor: float,
    scheduler_delay_threshold_s: float,
) -> str:
    lines = ["# Month 2 Operating-Point Analysis", "", "## Findings", ""]
    analyzed: list[tuple[str, str, str, list[dict[str, str]]]] = []
    for label, raw_rows in runs:
        dimension, dimension_label = varying_dimension(raw_rows)
        rows = sorted_rows(raw_rows, dimension)
        boundary, reasons = detect_boundary(
            rows,
            dimension,
            latency_factor,
            scheduler_delay_threshold_s,
        )
        if boundary is not None:
            boundary_value = format_value(boundary.get(dimension))
            boundary_text = (
                f"degradation candidate at {dimension_label.lower()} {boundary_value}"
            )
            reason_text = "; ".join(reasons)
        elif dimension != "workload":
            boundary_text = "no degradation trigger detected in the tested range"
            reason_text = ""
        else:
            boundary_text = f"controlled {dimension_label.lower()} comparison"
            reason_text = ""
        diagnosis = diagnose_bottleneck(rows)
        finding = f"- **{label}:** {boundary_text}. {diagnosis}"
        if reason_text:
            finding += f" Trigger: {reason_text}."
        lines.append(finding)
        analyzed.append((label, dimension_label, diagnosis, rows))

    lines.extend(["", "## Detailed results", ""])
    for label, dimension_label, diagnosis, rows in analyzed:
        lines.extend(
            [
                f"### {label}",
                "",
                f"Primary comparison: {dimension_label}. {diagnosis}",
                "",
                *render_table(rows),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation rules", "",
            f"- A p99 latency increase of at least {latency_factor:.2f}x versus the first row marks a degradation candidate.",
            "- Any error rate above zero also marks a degradation candidate.",
            f"- Open-loop runs with scheduler-delay p95 above {scheduler_delay_threshold_s:.3f}s are load-generator-limited and should be rerun with more client capacity.",
            "- Prompt-token targets are generator estimates; use `prompt_tokens_avg` from server usage for the actual comparison.",
            "- Output-token targets are prompt instructions plus API caps; use `output_tokens_avg` to verify that each stage reached its intended length.",
            "- Cost metrics require `--gpu-hourly-cost-usd` or `GPU_HOURLY_COST_USD` and represent effective GPU runtime cost, not full infrastructure cost.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Benchmark run directories to analyze")
    parser.add_argument("--out", default="reports/report-02-results.md")
    parser.add_argument("--latency-explosion-factor", type=float, default=2.0)
    parser.add_argument("--scheduler-delay-threshold-s", type=float, default=0.1)
    args = parser.parse_args()

    if args.latency_explosion_factor <= 1:
        raise ValueError("latency-explosion-factor must be > 1")
    if args.scheduler_delay_threshold_s < 0:
        raise ValueError("scheduler-delay-threshold-s must be >= 0")

    runs = [load_run(Path(run_dir)) for run_dir in args.run_dirs]
    report = build_report(
        runs,
        args.latency_explosion_factor,
        args.scheduler_delay_threshold_s,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
