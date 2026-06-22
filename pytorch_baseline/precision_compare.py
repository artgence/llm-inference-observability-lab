#!/usr/bin/env python3
"""Run isolated FP16/BF16 Hugging Face generation cases and compare results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import DEFAULT_MODEL_ID, DEFAULT_REVISION, parse_int_list, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtypes", default="fp16,bf16")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--prompt-tokens", default="512,2048")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--force-output-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="benchmarks/pytorch/precision_compare.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ratio(candidate: Any, baseline: Any) -> float | None:
    try:
        baseline_value = float(baseline)
        if baseline_value == 0:
            return None
        return float(candidate) / baseline_value
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    dtypes = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    allowed = {"fp16", "bf16", "fp32"}
    if not dtypes or any(dtype not in allowed for dtype in dtypes):
        raise ValueError(f"dtypes must come from {sorted(allowed)}")
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_targets = parse_int_list(args.prompt_tokens)
    if args.max_new_tokens < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid token, warmup, or repetition count")

    output_path = Path(args.out)
    case_dir = output_path.parent / "precision_cases"
    script = Path(__file__).with_name("hf_generate.py")
    commands: list[tuple[str, Path, list[str]]] = []
    for dtype in dtypes:
        case_out = case_dir / f"hf_generate_{dtype}.json"
        command = [
            sys.executable,
            str(script),
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--device",
            args.device,
            "--dtype",
            dtype,
            "--batch-sizes",
            ",".join(str(value) for value in batch_sizes),
            "--prompt-tokens",
            ",".join(str(value) for value in prompt_targets),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--warmups",
            str(args.warmups),
            "--repetitions",
            str(args.repetitions),
            "--out",
            str(case_out),
        ]
        if not args.force_output_length:
            command.append("--no-force-output-length")
        if args.trust_remote_code:
            command.append("--trust-remote-code")
        commands.append((dtype, case_out, command))

    if args.dry_run:
        print(
            json.dumps(
                {"commands": [command for _, _, command in commands]}, indent=2
            )
        )
        return 0

    runs: list[dict[str, Any]] = []
    for dtype, case_out, command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        record: dict[str, Any] = {
            "dtype": dtype,
            "command": command,
            "returncode": result.returncode,
            "result_file": str(case_out),
        }
        if result.returncode == 0 and case_out.exists():
            record["status"] = "completed"
            record["result"] = json.loads(case_out.read_text(encoding="utf-8"))
        else:
            record["status"] = "unsupported_or_failed"
            record["stderr"] = result.stderr[-4000:]
        runs.append(record)

    comparisons: list[dict[str, Any]] = []
    completed = [run for run in runs if run["status"] == "completed"]
    if len(completed) >= 2:
        baseline = completed[0]
        baseline_cases = {
            (case["batch_size"], case["prompt_tokens_target"]): case
            for case in baseline["result"]["cases"]
            if case.get("status") == "completed"
        }
        for candidate in completed[1:]:
            for case in candidate["result"]["cases"]:
                if case.get("status") != "completed":
                    continue
                key = (case["batch_size"], case["prompt_tokens_target"])
                baseline_case = baseline_cases.get(key)
                if baseline_case is None:
                    continue
                comparisons.append(
                    {
                        "baseline_dtype": baseline["dtype"],
                        "candidate_dtype": candidate["dtype"],
                        "batch_size": key[0],
                        "prompt_tokens_target": key[1],
                        "latency_mean_ratio": ratio(
                            case.get("latency_mean_s"),
                            baseline_case.get("latency_mean_s"),
                        ),
                        "throughput_mean_ratio": ratio(
                            case.get("output_tokens_per_sec_mean"),
                            baseline_case.get("output_tokens_per_sec_mean"),
                        ),
                        "peak_allocated_ratio": ratio(
                            case.get("max_allocated_bytes_max"),
                            baseline_case.get("max_allocated_bytes_max"),
                        ),
                    }
                )
    payload = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "runs": runs,
        "comparisons": comparisons,
        "note": "Each dtype runs in a fresh process so CUDA allocator state does not leak between cases.",
    }
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "wrote": str(output_path),
                "run_status": {
                    run["dtype"]: run["status"] for run in runs
                },
                "comparison_count": len(comparisons),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
