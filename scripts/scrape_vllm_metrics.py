#!/usr/bin/env python3
"""Scrape raw Prometheus metrics from a running vLLM server."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
import urllib.request
from pathlib import Path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def scrape(metrics_url: str, timeout: float) -> str:
    with urllib.request.urlopen(metrics_url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    metrics_url = args.base_url.rstrip("/") + "/metrics"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    deadline = time.perf_counter() + args.duration
    with output.open("a", encoding="utf-8") as handle:
        while True:
            handle.write(f"# scrape_at: {utc_now()}\n")
            handle.write(scrape(metrics_url, args.timeout))
            handle.write("\n")
            handle.flush()
            if args.duration <= 0 or time.perf_counter() >= deadline:
                break
            time.sleep(max(args.interval, 1.0))

    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
