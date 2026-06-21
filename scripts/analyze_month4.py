#!/usr/bin/env python3
"""Combine Month 4 serving-knob benchmark runs into a comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DISPLAY_COLUMNS = [
    "server_config_label",
    "workload",
    "analysis_group",
    "concurrency",
    "prompt_tokens_target",
    "prompt_tokens_target_min",
    "prompt_tokens_target_max",
    "prefix_mode",
    "admission_control_action",
    "admission_rejected_count",
    "latency_p99_s",
    "ttft_p95_s",
    "tpot_p95_s",
    "requests_per_sec",
    "output_tokens_per_sec",
    "vllm_requests_waiting_max",
    "vllm_kv_cache_usage_pct_max",
    "vllm_prefix_cache_hit_rate",
    "vllm_preemptions",
    "gpu_memory_utilization_pct_max",
    "error_rate",
    "cost_per_1m_total_tokens_usd",
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


def ratio_text(candidate: Any, baseline: Any, lower_is_better: bool) -> str:
    candidate_value = safe_float(candidate)
    baseline_value = safe_float(baseline)
    if candidate_value is None or baseline_value is None or baseline_value == 0:
        return "n/a"
    change_pct = (candidate_value / baseline_value - 1) * 100
    direction = "improvement" if (change_pct < 0) == lower_is_better else "regression"
    return f"{abs(change_pct):.1f}% {direction}"


def load_run(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metadata_path = run_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    label = metadata.get("server_config_label") or run_dir.name
    for row in rows:
        row["server_config_label"] = str(label)
        row["run_id"] = str(metadata.get("run_id", run_dir.name))
        row["model"] = str(metadata.get("model", "unknown"))
    return rows


def render_table(rows: list[dict[str, str]]) -> list[str]:
    columns = [column for column in DISPLAY_COLUMNS if any(row.get(column) for row in rows)]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    return lines


def prefix_findings(rows: list[dict[str, str]]) -> list[str]:
    by_variant: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("analysis_group") == "prefix_cache" and row.get("prefix_mode"):
            by_variant[row["server_config_label"]][row["prefix_mode"]] = row
    findings: list[str] = []
    for variant, modes in sorted(by_variant.items()):
        if "shared" not in modes or "unique" not in modes:
            continue
        shared = modes["shared"]
        unique = modes["unique"]
        findings.append(
            f"- **{variant}:** shared versus unique prefix TTFT p95: "
            f"{ratio_text(shared.get('ttft_p95_s'), unique.get('ttft_p95_s'), True)}; "
            "output throughput: "
            f"{ratio_text(shared.get('output_tokens_per_sec'), unique.get('output_tokens_per_sec'), False)}; "
            f"shared-prefix cache hit rate: {fmt(shared.get('vllm_prefix_cache_hit_rate'))}."
        )
    return findings


def cross_variant_findings(rows: list[dict[str, str]]) -> list[str]:
    by_workload: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_workload[row.get("workload", "unknown")].append(row)
    findings: list[str] = []
    for workload, workload_rows in sorted(by_workload.items()):
        if len(workload_rows) < 2:
            continue
        baseline = workload_rows[0]
        for candidate in workload_rows[1:]:
            findings.append(
                f"- **{workload}: {candidate['server_config_label']} vs "
                f"{baseline['server_config_label']}:** TTFT p95 "
                f"{ratio_text(candidate.get('ttft_p95_s'), baseline.get('ttft_p95_s'), True)}; "
                "output throughput "
                f"{ratio_text(candidate.get('output_tokens_per_sec'), baseline.get('output_tokens_per_sec'), False)}."
            )
    return findings


def build_report(rows: list[dict[str, str]]) -> str:
    findings = prefix_findings(rows) + cross_variant_findings(rows)
    if not findings:
        findings = [
            "- Run at least two server variants for a cross-configuration comparison; raw results are shown below."
        ]
    lines = [
        "# Report 04: Practical vLLM Serving Knobs and Capacity Boundaries",
        "",
        "## Findings",
        "",
        *findings,
        "",
        "## Results",
        "",
        *render_table(rows),
        "",
        "## Interpretation",
        "",
        "- Prefix caching should primarily improve prefill/TTFT for repeated prefixes; it should not materially improve decode-heavy TPOT.",
        "- Treat high scheduler delay as a client-capacity problem before attributing latency to vLLM.",
        "- Admission rejections are intentional policy outcomes and must be separated from server failures and OOMs.",
        "- A serving knob is useful only if its latency or throughput benefit does not create unacceptable errors, KV-cache pressure, or cost.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Benchmark run directories, baseline first")
    parser.add_argument(
        "--out",
        default="reports/report-04-results.md",
        help="Output Markdown report",
    )
    args = parser.parse_args()
    rows = [row for run_dir in args.run_dirs for row in load_run(Path(run_dir))]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
