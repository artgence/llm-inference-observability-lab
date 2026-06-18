#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI-compatible chat endpoint with streaming TTFT capture."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


VOCAB = [
    "capacity",
    "latency",
    "throughput",
    "queueing",
    "prefill",
    "decode",
    "memory",
    "tokens",
    "batching",
    "telemetry",
    "timeout",
    "saturation",
    "utilization",
    "cache",
    "request",
    "service",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def make_run_id() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def generated_prompt(word_count: int, request_index: int) -> str:
    words = [VOCAB[i % len(VOCAB)] for i in range(word_count)]
    return (
        "You are evaluating an LLM inference service. Use the context below to "
        "identify operational risks and summarize them clearly.\n\n"
        f"Request id: {request_index}\n"
        "Context:\n"
        + " ".join(words)
        + "\n\nQuestion: Summarize the likely bottlenecks in three concise bullets."
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_error(message: str, status_code: int | None = None) -> str:
    text = message.lower()
    if status_code == 429 or "rejected" in text or "rate limit" in text:
        return "rejected"
    if "out of memory" in text or "cuda oom" in text or "cuda error" in text:
        return "oom"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "operation not permitted" in text or "connection refused" in text or "urlopen error" in text:
        return "network_error"
    if status_code is not None and status_code >= 500:
        return "server_error"
    if status_code is not None and status_code >= 400:
        return "client_error"
    return "error"


class GpuSampler:
    def __init__(self, output_path: Path, interval_seconds: float) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.available:
            self.output_path.with_name("gpu_metrics_unavailable.txt").write_text(
                "nvidia-smi was not found. Run this benchmark in an NVIDIA GPU environment "
                "to collect GPU memory and utilization samples.\n",
                encoding="utf-8",
            )
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "collected_at",
                "gpu_index",
                "gpu_name",
                "memory_used_mb",
                "memory_total_mb",
                "gpu_utilization_pct",
            ])
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval_seconds + 2)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self._sample_once()
            self.stop_event.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return

        rows = []
        collected_at = utc_now()
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            rows.append([collected_at, *parts[:5]])

        if rows:
            with self.output_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerows(rows)


def post_streaming_chat(
    endpoint: str,
    api_key: str,
    model: str,
    workload: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    prompt = workload.get("prompt") or generated_prompt(int(workload.get("prompt_words", 256)), request_index)
    timeout = float(workload.get("timeout_seconds", 120))
    started_perf = time.perf_counter()
    started_at = utc_now()
    ttft_s: float | None = None
    token_event_times: list[float] = []
    content_parts: list[str] = []
    usage: dict[str, Any] = {}
    status_code: int | None = None

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(workload.get("max_tokens", 128)),
        "temperature": float(workload.get("temperature", 0.0)),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(workload.get("extra_body", {}))

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.split("data:", 1)[1].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    now = time.perf_counter()
                    if ttft_s is None:
                        ttft_s = now - started_perf
                    token_event_times.append(now)
                    content_parts.append(text)
    except urllib.error.HTTPError as exc:
        ended_perf = time.perf_counter()
        body = exc.read().decode("utf-8", "replace")
        error_type = classify_error(body, exc.code)
        return {
            "request_index": request_index,
            "started_at": started_at,
            "ended_at": utc_now(),
            "success": False,
            "status_code": exc.code,
            "error_type": error_type,
            "error_message": body[:1000],
            "latency_s": ended_perf - started_perf,
            "timeout": error_type == "timeout",
            "oom": error_type == "oom",
            "rejected": error_type == "rejected",
        }
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        ended_perf = time.perf_counter()
        message = str(exc)
        error_type = classify_error(message)
        return {
            "request_index": request_index,
            "started_at": started_at,
            "ended_at": utc_now(),
            "success": False,
            "status_code": status_code,
            "error_type": error_type,
            "error_message": message[:1000],
            "latency_s": ended_perf - started_perf,
            "timeout": error_type == "timeout",
            "oom": error_type == "oom",
            "rejected": error_type == "rejected",
        }
    except Exception as exc:  # noqa: BLE001 - request records should retain unexpected failures.
        ended_perf = time.perf_counter()
        message = str(exc)
        error_type = classify_error(message)
        return {
            "request_index": request_index,
            "started_at": started_at,
            "ended_at": utc_now(),
            "success": False,
            "status_code": status_code,
            "error_type": error_type,
            "error_message": message[:1000],
            "latency_s": ended_perf - started_perf,
            "timeout": error_type == "timeout",
            "oom": error_type == "oom",
            "rejected": error_type == "rejected",
        }

    ended_perf = time.perf_counter()
    ended_at = utc_now()
    output_text = "".join(content_parts)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    token_source = "usage"

    if not isinstance(prompt_tokens, int):
        prompt_tokens = estimate_tokens(prompt)
        token_source = "estimated"
    if not isinstance(completion_tokens, int):
        completion_tokens = estimate_tokens(output_text) if output_text else 0
        token_source = "estimated"
    if not isinstance(total_tokens, int):
        total_tokens = prompt_tokens + completion_tokens

    tpot_s: float | None = None
    if ttft_s is not None and completion_tokens > 1:
        tpot_s = max(0.0, (ended_perf - started_perf - ttft_s) / (completion_tokens - 1))

    inter_token_latency_s: float | None = None
    if len(token_event_times) > 1:
        intervals = [
            token_event_times[i] - token_event_times[i - 1]
            for i in range(1, len(token_event_times))
        ]
        inter_token_latency_s = sum(intervals) / len(intervals)

    return {
        "request_index": request_index,
        "started_at": started_at,
        "ended_at": ended_at,
        "success": True,
        "status_code": status_code,
        "latency_s": ended_perf - started_perf,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "inter_token_latency_s": inter_token_latency_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_count_source": token_source,
        "observed_stream_chunks": len(token_event_times),
        "timeout": False,
        "oom": False,
        "rejected": False,
        "error_type": None,
        "error_message": None,
    }


def load_gpu_summary(gpu_path: Path) -> dict[str, float | None]:
    if not gpu_path.exists():
        return {"gpu_memory_used_mb_max": None, "gpu_utilization_pct_avg": None}
    memory_values: list[float] = []
    util_values: list[float] = []
    with gpu_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            memory = safe_float(row.get("memory_used_mb"))
            util = safe_float(row.get("gpu_utilization_pct"))
            if memory is not None:
                memory_values.append(memory)
            if util is not None:
                util_values.append(util)
    return {
        "gpu_memory_used_mb_max": max(memory_values) if memory_values else None,
        "gpu_utilization_pct_avg": sum(util_values) / len(util_values) if util_values else None,
    }


def summarize_workload(
    workload_name: str,
    results: list[dict[str, Any]],
    wall_time_s: float,
    gpu_summary: dict[str, float | None],
) -> dict[str, Any]:
    successes = [row for row in results if row.get("success")]
    latencies = [float(row["latency_s"]) for row in successes if row.get("latency_s") is not None]
    ttfts = [float(row["ttft_s"]) for row in successes if row.get("ttft_s") is not None]
    tpots = [float(row["tpot_s"]) for row in successes if row.get("tpot_s") is not None]
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in successes)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in successes)
    request_count = len(results)
    success_count = len(successes)

    return {
        "workload": workload_name,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": request_count - success_count,
        "timeout_count": sum(1 for row in results if row.get("timeout")),
        "oom_count": sum(1 for row in results if row.get("oom")),
        "rejected_count": sum(1 for row in results if row.get("rejected")),
        "error_rate": (request_count - success_count) / request_count if request_count else None,
        "timeout_rate": sum(1 for row in results if row.get("timeout")) / request_count if request_count else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "latency_p99_s": percentile(latencies, 0.99),
        "ttft_p50_s": percentile(ttfts, 0.50),
        "ttft_p95_s": percentile(ttfts, 0.95),
        "ttft_p99_s": percentile(ttfts, 0.99),
        "tpot_p50_s": percentile(tpots, 0.50),
        "tpot_p95_s": percentile(tpots, 0.95),
        "tpot_p99_s": percentile(tpots, 0.99),
        "requests_per_sec": success_count / wall_time_s if wall_time_s > 0 else None,
        "output_tokens_per_sec": output_tokens / wall_time_s if wall_time_s > 0 else None,
        "total_tokens_per_sec": total_tokens / wall_time_s if wall_time_s > 0 else None,
        "gpu_memory_used_mb_max": gpu_summary.get("gpu_memory_used_mb_max"),
        "gpu_utilization_pct_avg": gpu_summary.get("gpu_utilization_pct_avg"),
        "wall_time_s": wall_time_s,
    }


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "workload",
        "success_count",
        "error_count",
        "timeout_count",
        "oom_count",
        "rejected_count",
        "error_rate",
        "timeout_rate",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "ttft_p50_s",
        "tpot_p50_s",
        "requests_per_sec",
        "output_tokens_per_sec",
        "total_tokens_per_sec",
        "gpu_memory_used_mb_max",
        "gpu_utilization_pct_avg",
    ]
    lines = ["# Benchmark Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_workload(config: dict[str, Any]) -> None:
    if "runs" not in config or not isinstance(config["runs"], list) or not config["runs"]:
        raise ValueError("workload must contain a non-empty 'runs' list")
    for index, run in enumerate(config["runs"]):
        for key in ("name", "request_count", "concurrency", "max_tokens"):
            if key not in run:
                raise ValueError(f"run {index} is missing required key '{key}'")
        if int(run["request_count"]) < 1:
            raise ValueError(f"run {run['name']} must have request_count >= 1")
        if int(run["concurrency"]) < 1:
            raise ValueError(f"run {run['name']} must have concurrency >= 1")


def dry_run(config: dict[str, Any], model: str, base_url: str) -> None:
    plan = {
        "model": model,
        "base_url": base_url,
        "runs": [
            {
                "name": run["name"],
                "request_count": int(run["request_count"]),
                "concurrency": int(run["concurrency"]),
                "prompt_words": int(run.get("prompt_words", 256)),
                "max_tokens": int(run["max_tokens"]),
                "timeout_seconds": float(run.get("timeout_seconds", 120)),
                "example_prompt_estimated_tokens": estimate_tokens(
                    generated_prompt(int(run.get("prompt_words", 256)), 0)
                ),
            }
            for run in config["runs"]
        ],
    }
    print(json.dumps(plan, indent=2))


def run_benchmark(args: argparse.Namespace) -> int:
    config = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    validate_workload(config)

    model = os.environ.get("SERVED_MODEL_NAME") or os.environ.get("MODEL_ID") or config.get("model")
    if not model:
        raise ValueError("model must be set in workload or MODEL_ID/SERVED_MODEL_NAME")
    base_url = os.environ.get("VLLM_BASE_URL") or config.get("base_url", "http://localhost:8000")
    api_key = os.environ.get("OPENAI_API_KEY") or config.get("api_key", "EMPTY")
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"

    if args.dry_run:
        dry_run(config, model, base_url.rstrip("/"))
        return 0

    run_id = args.run_id or make_run_id()
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    request_log_path = run_dir / "requests.jsonl"
    gpu_path = run_dir / "gpu_metrics.csv"
    summary_csv_path = run_dir / "summary.csv"
    summary_md_path = run_dir / "summary.md"
    metadata_path = run_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": utc_now(),
                "model": model,
                "base_url": base_url.rstrip("/"),
                "workload_file": str(Path(args.workload)),
                "workload_description": config.get("description"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sampler = GpuSampler(gpu_path, args.gpu_sample_interval)
    sampler.start()
    summary_rows: list[dict[str, Any]] = []

    try:
        with request_log_path.open("w", encoding="utf-8") as request_log:
            for workload in config["runs"]:
                workload_name = workload["name"]
                request_count = int(workload["request_count"])
                concurrency = int(workload["concurrency"])
                workload_results: list[dict[str, Any]] = []
                workload_started = time.perf_counter()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = [
                        executor.submit(
                            post_streaming_chat,
                            endpoint,
                            api_key,
                            model,
                            workload,
                            index,
                        )
                        for index in range(request_count)
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        result.update(
                            {
                                "run_id": run_id,
                                "workload": workload_name,
                                "model": model,
                                "concurrency": concurrency,
                                "requested_max_tokens": int(workload["max_tokens"]),
                                "prompt_words": int(workload.get("prompt_words", 256)),
                            }
                        )
                        workload_results.append(result)
                        request_log.write(json.dumps(result) + "\n")
                        request_log.flush()

                wall_time_s = time.perf_counter() - workload_started
                gpu_summary = load_gpu_summary(gpu_path)
                summary_rows.append(
                    summarize_workload(workload_name, workload_results, wall_time_s, gpu_summary)
                )
    finally:
        sampler.stop()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ended_at"] = utc_now()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_summary_csv(summary_csv_path, summary_rows)
    write_summary_md(summary_md_path, summary_rows)
    print(f"Wrote run artifacts to {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="workloads/month1_baseline.json")
    parser.add_argument("--out-dir", default="benchmarks")
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
