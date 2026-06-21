#!/usr/bin/env python3
"""Render a Markdown table from a benchmark run directory."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "workload",
    "load_mode",
    "arrival_pattern",
    "burst_size",
    "concurrency",
    "prompt_tokens_target",
    "max_tokens",
    "output_tokens_target",
    "target_request_rate_rps",
    "success_count",
    "error_count",
    "timeout_count",
    "oom_count",
    "rejected_count",
    "latency_p50_s",
    "latency_p95_s",
    "latency_p99_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "tpot_p50_s",
    "tpot_p95_s",
    "prompt_tokens_avg",
    "output_tokens_avg",
    "requests_per_sec",
    "output_tokens_per_sec",
    "total_tokens_per_sec",
    "gpu_memory_used_mb_max",
    "gpu_memory_utilization_pct_max",
    "gpu_utilization_pct_avg",
    "cost_per_1m_output_tokens_usd",
    "cost_per_1m_total_tokens_usd",
]


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}"


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_markdown(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = ["# Benchmark Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Benchmark run directory containing summary.csv")
    parser.add_argument("--out", help="Optional output Markdown path. Defaults to <run_dir>/summary.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv}")

    rows = read_summary(summary_csv)
    columns = [column for column in DEFAULT_COLUMNS if rows and column in rows[0]]
    markdown = render_markdown(rows, columns)
    out_path = Path(args.out) if args.out else run_dir / "summary.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
