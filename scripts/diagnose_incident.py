#!/usr/bin/env python3
"""Generate an incident-style diagnosis note from a benchmark run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


INCIDENT_GUIDANCE = {
    "traffic_burst": {
        "title": "Traffic Burst",
        "mitigation": "Reduce admission rate, apply bounded queueing, and shed or retry excess traffic with jitter.",
        "alert": "Alert on sustained waiting requests, TTFT p95, and failure rate during an arrival spike.",
        "control": "Enforce rate limits, maximum queued requests, and autoscaling or replica headroom.",
    },
    "long_prompt_storm": {
        "title": "Long Prompt Storm",
        "mitigation": "Throttle long prompts separately, cap prompt length, and use chunked prefill where appropriate.",
        "alert": "Alert on prompt-token rate, prefill/TTFT p95, waiting requests, and KV-cache growth.",
        "control": "Apply prompt-length-aware admission control and per-tier token budgets.",
    },
    "memory_pressure": {
        "title": "KV-Cache Memory Pressure",
        "mitigation": "Reduce concurrency or sequence length, lower admission rate, and restart only after preserving failure evidence.",
        "alert": "Alert on KV-cache usage above 90%, waiting requests, preemptions, and any OOM classification.",
        "control": "Set safe concurrency/token budgets and reserve GPU-memory headroom below the tested failure boundary.",
    },
}


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, suffix: str = "") -> str:
    number = safe_float(value)
    return f"{number:.3f}{suffix}" if number is not None else "n/a"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worst_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(
        rows,
        key=lambda row: (
            safe_float(row.get("error_rate")) or 0,
            safe_float(row.get("latency_p99_s")) or 0,
            safe_float(row.get("ttft_p95_s")) or 0,
        ),
    )


def likely_bottleneck(incident_type: str, row: dict[str, str]) -> str:
    if (safe_float(row.get("oom_count")) or 0) > 0:
        return "GPU memory exhaustion; OOM failures were observed."
    kv_usage = safe_float(row.get("vllm_kv_cache_usage_pct_max"))
    if kv_usage is not None and kv_usage >= 90:
        return f"KV-cache pressure; peak vLLM cache usage reached {kv_usage:.1f}%."
    waiting = safe_float(row.get("vllm_requests_waiting_max")) or 0
    if waiting > 0:
        return f"Scheduler queueing; waiting requests peaked at {waiting:.0f}."
    if incident_type == "long_prompt_storm":
        return "Prefill pressure from long prompts, inferred from the incident workload shape."
    if incident_type == "memory_pressure":
        return "KV-cache or GPU-memory pressure, pending confirmation from server metrics."
    return "Queueing caused by burst arrival shape, pending confirmation from TTFT and waiting-request metrics."


def build_note(run_dir: Path, incident_type: str) -> str:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    rows = load_csv(summary_path)
    if not rows:
        raise ValueError(f"No summary rows in {summary_path}")
    row = worst_row(rows)
    guidance = INCIDENT_GUIDANCE[incident_type]
    metadata_path = run_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    total_requests = sum(int(safe_float(item.get("request_count")) or 0) for item in rows)
    total_errors = sum(int(safe_float(item.get("error_count")) or 0) for item in rows)
    total_timeouts = sum(int(safe_float(item.get("timeout_count")) or 0) for item in rows)
    total_ooms = sum(int(safe_float(item.get("oom_count")) or 0) for item in rows)
    total_rejected = sum(int(safe_float(item.get("rejected_count")) or 0) for item in rows)
    impact_pct = total_errors / total_requests if total_requests else 0

    metrics = [
        f"Worst workload: `{row.get('workload', 'unknown')}`",
        f"Latency p99: {fmt(row.get('latency_p99_s'), 's')}",
        f"TTFT p95: {fmt(row.get('ttft_p95_s'), 's')}",
        f"TPOT p95: {fmt(row.get('tpot_p95_s'), 's')}",
        f"Requests running max: {fmt(row.get('vllm_requests_running_max'))}",
        f"Requests waiting max: {fmt(row.get('vllm_requests_waiting_max'))}",
        f"KV-cache usage max: {fmt(row.get('vllm_kv_cache_usage_pct_max'), '%')}",
        f"GPU memory usage max: {fmt(row.get('gpu_memory_utilization_pct_max'), '%')}",
        f"GPU utilization average: {fmt(row.get('gpu_utilization_pct_avg'), '%')}",
        f"Output throughput: {fmt(row.get('output_tokens_per_sec'), ' tokens/s')}",
    ]

    return "\n".join(
        [
            f"# Incident: {guidance['title']}",
            "",
            f"Run ID: `{metadata.get('run_id', run_dir.name)}`",
            f"Model: `{metadata.get('model', 'unknown')}`",
            "",
            "## Symptom",
            "",
            f"The most degraded stage was `{row.get('workload', 'unknown')}` with p99 latency {fmt(row.get('latency_p99_s'), 's')} and error rate {fmt((safe_float(row.get('error_rate')) or 0) * 100, '%')}.",
            "",
            "## Impact",
            "",
            f"{total_errors} of {total_requests} requests failed ({impact_pct:.2%}); timeouts={total_timeouts}, OOMs={total_ooms}, rejected={total_rejected}.",
            "",
            "## Metrics observed",
            "",
            *[f"- {metric}" for metric in metrics],
            "",
            "## Likely bottleneck",
            "",
            likely_bottleneck(incident_type, row),
            "",
            "## Mitigation",
            "",
            guidance["mitigation"],
            "",
            "## What alert should exist",
            "",
            guidance["alert"],
            "",
            "## What config/control would prevent this",
            "",
            guidance["control"],
            "",
            "## Evidence quality",
            "",
            f"vLLM metric sampling errors for the worst workload: {fmt(row.get('vllm_metrics_sample_errors'))}. Treat queue/KV conclusions as provisional if this value is non-zero or the metrics are unavailable.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--incident-type", choices=sorted(INCIDENT_GUIDANCE), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    note = build_note(Path(args.run_dir), args.incident_type)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(note, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
