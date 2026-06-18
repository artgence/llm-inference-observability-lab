#!/usr/bin/env python3
"""Poll nvidia-smi and write GPU memory/utilization samples to CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shutil
import subprocess
import time
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sample_once() -> list[list[str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    collected_at = utc_now()
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append([collected_at, *parts[:5]])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi was not found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + args.duration
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "collected_at",
            "gpu_index",
            "gpu_name",
            "memory_used_mb",
            "memory_total_mb",
            "gpu_utilization_pct",
        ])
        while time.perf_counter() < deadline:
            writer.writerows(sample_once())
            handle.flush()
            time.sleep(args.interval)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
