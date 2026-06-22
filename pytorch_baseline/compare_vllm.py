#!/usr/bin/env python3
"""Join PyTorch generation results with a matching vLLM summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    parsed = number(value)
    return f"{parsed:.3f}" if parsed is not None else "n/a"


def ratio(candidate: Any, baseline: Any) -> float | None:
    candidate_value = number(candidate)
    baseline_value = number(baseline)
    if candidate_value is None or baseline_value in (None, 0):
        return None
    return candidate_value / baseline_value


def delta_pct(candidate: Any, baseline: Any) -> float | None:
    value = ratio(candidate, baseline)
    return (value - 1) * 100 if value is not None else None


def load_vllm_summary(path: Path) -> list[dict[str, str]]:
    summary = path / "summary.csv" if path.is_dir() else path
    if not summary.exists():
        raise FileNotFoundError(f"Missing {summary}")
    with summary.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rows(
    pytorch_payload: dict[str, Any], vllm_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    vllm_index: dict[tuple[int, int], dict[str, str]] = {}
    for row in vllm_rows:
        prompt = number(row.get("prompt_tokens_target"))
        concurrency = number(row.get("concurrency"))
        if prompt is not None and concurrency is not None:
            vllm_index[(int(prompt), int(concurrency))] = row

    joined: list[dict[str, Any]] = []
    for case in pytorch_payload.get("cases", []):
        if case.get("status") != "completed":
            continue
        key = (int(case["prompt_tokens_target"]), int(case["batch_size"]))
        vllm = vllm_index.get(key, {})
        joined.append(
            {
                "prompt_tokens_target": key[0],
                "pytorch_batch_size": key[1],
                "vllm_concurrency": key[1],
                "pytorch_prompt_tokens_actual": case.get("prompt_tokens_actual"),
                "vllm_prompt_tokens_avg": vllm.get("prompt_tokens_avg"),
                "prompt_token_count_delta_pct": delta_pct(
                    vllm.get("prompt_tokens_avg"),
                    case.get("prompt_tokens_actual"),
                ),
                "pytorch_latency_p50_s": case.get("latency_p50_s"),
                "pytorch_latency_p95_s": case.get("latency_p95_s"),
                "vllm_latency_p50_s": vllm.get("latency_p50_s"),
                "vllm_latency_p95_s": vllm.get("latency_p95_s"),
                "pytorch_output_tokens_per_sec": case.get(
                    "output_tokens_per_sec_mean"
                ),
                "vllm_output_tokens_per_sec": vllm.get("output_tokens_per_sec"),
                "vllm_to_pytorch_throughput_ratio": ratio(
                    vllm.get("output_tokens_per_sec"),
                    case.get("output_tokens_per_sec_mean"),
                ),
                "pytorch_peak_allocated_bytes": case.get(
                    "max_allocated_bytes_max"
                ),
                "vllm_gpu_memory_used_mb_max": vllm.get(
                    "gpu_memory_used_mb_max"
                ),
                "vllm_error_rate": vllm.get("error_rate"),
            }
        )
    return joined


def render_markdown(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0]) if rows else []
    lines = [
        "# PyTorch vs vLLM Comparison",
        "",
        "PyTorch batch size and vLLM request concurrency are paired by shape but are not identical scheduling semantics. PyTorch latency is in-process; vLLM latency is client end-to-end.",
        "",
    ]
    if not rows:
        lines.append("No matching rows found.")
        return "\n".join(lines) + "\n"
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    lines.extend(
        [
            "",
            "Reject direct conclusions when actual prompt/output token counts, model revision, dtype, or output stopping behavior differ materially.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pytorch_result", help="hf_generate JSON result")
    parser.add_argument("vllm_result", help="vLLM run directory or summary.csv")
    parser.add_argument(
        "--out",
        default="reports/report-02.5-comparison-results.md",
    )
    args = parser.parse_args()
    pytorch_payload = json.loads(
        Path(args.pytorch_result).read_text(encoding="utf-8")
    )
    rows = build_rows(pytorch_payload, load_vllm_summary(Path(args.vllm_result)))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(rows), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
