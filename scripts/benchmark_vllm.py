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


DEFAULT_RESPONSE_INSTRUCTION = "Summarize the likely bottlenecks in three concise bullets."


def generated_prompt(
    word_count: int,
    request_index: int,
    response_instruction: str = DEFAULT_RESPONSE_INSTRUCTION,
) -> str:
    words = [VOCAB[i % len(VOCAB)] for i in range(word_count)]
    return (
        "You are evaluating an LLM inference service. Use the context below to "
        "identify operational risks and summarize them clearly.\n\n"
        f"Request id: {request_index}\n"
        "Context:\n"
        + " ".join(words)
        + f"\n\nQuestion: {response_instruction}"
    )


def generated_prompt_for_token_target(
    token_count: int,
    request_index: int,
    response_instruction: str = DEFAULT_RESPONSE_INSTRUCTION,
) -> str:
    """Generate a deterministic prompt near a target using the harness's chars/token estimate."""
    prefix = (
        "You are evaluating an LLM inference service. Identify operational risks.\n\n"
        f"Request id: {request_index}\nContext:\n"
    )
    suffix = f"\n\nQuestion: {response_instruction}"
    target_chars = token_count * 4
    filler_chars = max(0, target_chars - len(prefix) - len(suffix))
    filler = ("the " * math.ceil(filler_chars / 4))[:filler_chars]
    return prefix + filler + suffix


def prompt_for_workload(workload: dict[str, Any], request_index: int) -> str:
    if workload.get("prompt"):
        return str(workload["prompt"])
    response_instruction = str(
        workload.get("response_instruction", DEFAULT_RESPONSE_INSTRUCTION)
    )
    if workload.get("prompt_tokens") is not None:
        return generated_prompt_for_token_target(
            int(workload["prompt_tokens"]),
            request_index,
            response_instruction,
        )
    return generated_prompt(
        int(workload.get("prompt_words", 256)),
        request_index,
        response_instruction,
    )


def scheduled_offset_for_request(workload: dict[str, Any], request_index: int) -> float:
    arrival_rate_rps = float(workload["arrival_rate_rps"])
    if workload.get("arrival_pattern", "steady") == "bursty":
        burst_size = int(workload["burst_size"])
        return (request_index // burst_size) * burst_size / arrival_rate_rps
    return request_index / arrival_rate_rps


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
        self.workload_name = "unassigned"
        self.workload_lock = threading.Lock()

    def set_workload(self, workload_name: str) -> None:
        with self.workload_lock:
            self.workload_name = workload_name

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
                "workload",
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
        with self.workload_lock:
            workload_name = self.workload_name
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            rows.append([collected_at, workload_name, *parts[:5]])

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
    workload_started_perf: float | None = None,
    scheduled_offset_s: float | None = None,
) -> dict[str, Any]:
    prompt = prompt_for_workload(workload, request_index)
    timeout = float(workload.get("timeout_seconds", 120))
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

    def scheduling_fields(ended_perf: float) -> dict[str, Any]:
        request_start_offset_s = (
            started_perf - workload_started_perf if workload_started_perf is not None else None
        )
        scheduler_delay_s = (
            max(0.0, request_start_offset_s - scheduled_offset_s)
            if request_start_offset_s is not None and scheduled_offset_s is not None
            else None
        )
        scheduled_latency_s = (
            ended_perf - workload_started_perf - scheduled_offset_s
            if workload_started_perf is not None and scheduled_offset_s is not None
            else None
        )
        return {
            "latency_origin": "request_send",
            "scheduled_offset_s": scheduled_offset_s,
            "request_start_offset_s": request_start_offset_s,
            "scheduler_delay_s": scheduler_delay_s,
            "scheduled_latency_s": scheduled_latency_s,
        }

    started_at = utc_now()
    started_perf = time.perf_counter()
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
            **scheduling_fields(ended_perf),
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
            **scheduling_fields(ended_perf),
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
            **scheduling_fields(ended_perf),
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
        **scheduling_fields(ended_perf),
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


def load_gpu_summary(gpu_path: Path, workload_name: str | None = None) -> dict[str, float | None]:
    if not gpu_path.exists():
        return {
            "gpu_memory_used_mb_max": None,
            "gpu_memory_utilization_pct_max": None,
            "gpu_utilization_pct_avg": None,
        }
    memory_values: list[float] = []
    memory_utilization_values: list[float] = []
    util_values: list[float] = []
    with gpu_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sampled_workload = row.get("workload")
            if workload_name is not None and sampled_workload and sampled_workload != workload_name:
                continue
            memory = safe_float(row.get("memory_used_mb"))
            memory_total = safe_float(row.get("memory_total_mb"))
            util = safe_float(row.get("gpu_utilization_pct"))
            if memory is not None:
                memory_values.append(memory)
                if memory_total is not None and memory_total > 0:
                    memory_utilization_values.append(memory / memory_total)
            if util is not None:
                util_values.append(util)
    return {
        "gpu_memory_used_mb_max": max(memory_values) if memory_values else None,
        "gpu_memory_utilization_pct_max": (
            max(memory_utilization_values) * 100 if memory_utilization_values else None
        ),
        "gpu_utilization_pct_avg": sum(util_values) / len(util_values) if util_values else None,
    }


def summarize_workload(
    workload_name: str,
    workload: dict[str, Any],
    results: list[dict[str, Any]],
    wall_time_s: float,
    gpu_summary: dict[str, float | None],
    gpu_hourly_cost_usd: float | None,
) -> dict[str, Any]:
    successes = [row for row in results if row.get("success")]
    latencies = [float(row["latency_s"]) for row in successes if row.get("latency_s") is not None]
    ttfts = [float(row["ttft_s"]) for row in successes if row.get("ttft_s") is not None]
    tpots = [float(row["tpot_s"]) for row in successes if row.get("tpot_s") is not None]
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in successes)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in successes)
    prompt_token_values = [
        int(row["prompt_tokens"])
        for row in successes
        if row.get("prompt_tokens") is not None
    ]
    output_token_values = [
        int(row["output_tokens"])
        for row in successes
        if row.get("output_tokens") is not None
    ]
    scheduler_delays = [
        float(row["scheduler_delay_s"])
        for row in results
        if row.get("scheduler_delay_s") is not None
    ]
    scheduled_latencies = [
        float(row["scheduled_latency_s"])
        for row in successes
        if row.get("scheduled_latency_s") is not None
    ]
    request_start_offsets = [
        float(row["request_start_offset_s"])
        for row in results
        if row.get("request_start_offset_s") is not None
    ]
    request_count = len(results)
    success_count = len(successes)
    load_mode = workload.get("load_mode", "closed_loop")
    target_request_rate_rps = safe_float(workload.get("arrival_rate_rps"))
    achieved_request_start_rate_rps: float | None = None
    if load_mode == "open_loop" and len(request_start_offsets) > 1:
        request_start_span_s = max(request_start_offsets) - min(request_start_offsets)
        arrival_rate_rps = float(workload["arrival_rate_rps"])
        burst_size = int(workload.get("burst_size", 1))
        final_arrival_count = request_count % burst_size or burst_size
        arrival_window_s = final_arrival_count / arrival_rate_rps
        observed_window_s = request_start_span_s + arrival_window_s
        nominal_window_s = request_count / arrival_rate_rps
        achieved_window_s = max(observed_window_s, nominal_window_s)
        if achieved_window_s > 0:
            achieved_request_start_rate_rps = request_count / achieved_window_s
    estimated_gpu_cost_usd = (
        gpu_hourly_cost_usd * wall_time_s / 3600 if gpu_hourly_cost_usd is not None else None
    )
    cost_per_1m_output_tokens_usd = (
        estimated_gpu_cost_usd * 1_000_000 / output_tokens
        if estimated_gpu_cost_usd is not None and output_tokens > 0
        else None
    )
    cost_per_1m_total_tokens_usd = (
        estimated_gpu_cost_usd * 1_000_000 / total_tokens
        if estimated_gpu_cost_usd is not None and total_tokens > 0
        else None
    )

    return {
        "workload": workload_name,
        "load_mode": load_mode,
        "arrival_pattern": (
            workload.get("arrival_pattern", "steady")
            if load_mode == "open_loop"
            else "closed_loop"
        ),
        "burst_size": int(workload["burst_size"]) if "burst_size" in workload else None,
        "concurrency": int(workload["concurrency"]) if "concurrency" in workload else None,
        "prompt_tokens_target": (
            int(workload["prompt_tokens"]) if "prompt_tokens" in workload else None
        ),
        "prompt_words": int(workload["prompt_words"]) if "prompt_words" in workload else None,
        "max_tokens": int(workload["max_tokens"]),
        "output_tokens_target": (
            int(workload["output_tokens_target"])
            if "output_tokens_target" in workload
            else None
        ),
        "target_request_rate_rps": target_request_rate_rps,
        "achieved_request_start_rate_rps": achieved_request_start_rate_rps,
        "scheduler_delay_p95_s": percentile(scheduler_delays, 0.95),
        "scheduled_latency_p95_s": percentile(scheduled_latencies, 0.95),
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
        "prompt_tokens_avg": (
            sum(prompt_token_values) / len(prompt_token_values) if prompt_token_values else None
        ),
        "output_tokens_avg": (
            sum(output_token_values) / len(output_token_values) if output_token_values else None
        ),
        "requests_per_sec": success_count / wall_time_s if wall_time_s > 0 else None,
        "output_tokens_per_sec": output_tokens / wall_time_s if wall_time_s > 0 else None,
        "total_tokens_per_sec": total_tokens / wall_time_s if wall_time_s > 0 else None,
        "gpu_memory_used_mb_max": gpu_summary.get("gpu_memory_used_mb_max"),
        "gpu_memory_utilization_pct_max": gpu_summary.get(
            "gpu_memory_utilization_pct_max"
        ),
        "gpu_utilization_pct_avg": gpu_summary.get("gpu_utilization_pct_avg"),
        "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
        "estimated_gpu_cost_usd": estimated_gpu_cost_usd,
        "cost_per_1m_output_tokens_usd": cost_per_1m_output_tokens_usd,
        "cost_per_1m_total_tokens_usd": cost_per_1m_total_tokens_usd,
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
        "load_mode",
        "arrival_pattern",
        "burst_size",
        "concurrency",
        "prompt_tokens_target",
        "max_tokens",
        "output_tokens_target",
        "target_request_rate_rps",
        "achieved_request_start_rate_rps",
        "scheduler_delay_p95_s",
        "scheduled_latency_p95_s",
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
        "gpu_hourly_cost_usd",
        "cost_per_1m_output_tokens_usd",
        "cost_per_1m_total_tokens_usd",
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
    names: set[str] = set()
    for index, run in enumerate(config["runs"]):
        for key in ("name", "request_count", "max_tokens"):
            if key not in run:
                raise ValueError(f"run {index} is missing required key '{key}'")
        if int(run["request_count"]) < 1:
            raise ValueError(f"run {run['name']} must have request_count >= 1")
        if int(run["max_tokens"]) < 1:
            raise ValueError(f"run {run['name']} must have max_tokens >= 1")
        if run["name"] in names:
            raise ValueError(f"duplicate workload name '{run['name']}'")
        names.add(run["name"])
        if "prompt_tokens" in run and int(run["prompt_tokens"]) < 1:
            raise ValueError(f"run {run['name']} must have prompt_tokens >= 1")
        if "output_tokens_target" in run and int(run["output_tokens_target"]) < 1:
            raise ValueError(f"run {run['name']} must have output_tokens_target >= 1")
        load_mode = run.get("load_mode", "closed_loop")
        if load_mode not in {"closed_loop", "open_loop"}:
            raise ValueError(f"run {run['name']} has unsupported load_mode '{load_mode}'")
        if load_mode == "closed_loop":
            if "concurrency" not in run:
                raise ValueError(f"closed-loop run {run['name']} must define concurrency")
            if int(run["concurrency"]) < 1:
                raise ValueError(f"run {run['name']} must have concurrency >= 1")
        else:
            if (
                safe_float(run.get("arrival_rate_rps")) is None
                or float(run["arrival_rate_rps"]) <= 0
            ):
                raise ValueError(f"open-loop run {run['name']} must have arrival_rate_rps > 0")
            if int(run.get("max_in_flight", 0)) < 1:
                raise ValueError(f"open-loop run {run['name']} must have max_in_flight >= 1")
            arrival_pattern = run.get("arrival_pattern", "steady")
            if arrival_pattern not in {"steady", "bursty"}:
                raise ValueError(
                    f"open-loop run {run['name']} has unsupported arrival_pattern "
                    f"'{arrival_pattern}'"
                )
            if arrival_pattern == "bursty" and int(run.get("burst_size", 0)) < 2:
                raise ValueError(f"bursty run {run['name']} must have burst_size >= 2")


def dry_run(config: dict[str, Any], model: str, base_url: str) -> None:
    plan = {
        "model": model,
        "base_url": base_url,
        "runs": [
            {
                "name": run["name"],
                "load_mode": run.get("load_mode", "closed_loop"),
                "arrival_pattern": (
                    run.get("arrival_pattern", "steady")
                    if run.get("load_mode", "closed_loop") == "open_loop"
                    else "closed_loop"
                ),
                "request_count": int(run["request_count"]),
                "concurrency": int(run["concurrency"]) if "concurrency" in run else None,
                "arrival_rate_rps": safe_float(run.get("arrival_rate_rps")),
                "burst_size": int(run["burst_size"]) if "burst_size" in run else None,
                "max_in_flight": int(run["max_in_flight"]) if "max_in_flight" in run else None,
                "prompt_tokens": int(run["prompt_tokens"]) if "prompt_tokens" in run else None,
                "prompt_words": int(run["prompt_words"]) if "prompt_words" in run else None,
                "max_tokens": int(run["max_tokens"]),
                "output_tokens_target": (
                    int(run["output_tokens_target"])
                    if "output_tokens_target" in run
                    else None
                ),
                "timeout_seconds": float(run.get("timeout_seconds", 120)),
                "extra_body_keys": sorted((run.get("extra_body") or {}).keys()),
                "example_prompt_estimated_tokens": estimate_tokens(
                    prompt_for_workload(run, 0)
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
    gpu_hourly_cost_usd = safe_float(
        getattr(args, "gpu_hourly_cost_usd", None)
        if getattr(args, "gpu_hourly_cost_usd", None) is not None
        else os.environ.get("GPU_HOURLY_COST_USD", config.get("gpu_hourly_cost_usd"))
    )
    if gpu_hourly_cost_usd is not None and gpu_hourly_cost_usd < 0:
        raise ValueError("gpu_hourly_cost_usd must be >= 0")

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
                "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
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
                sampler.set_workload(workload_name)
                request_count = int(workload["request_count"])
                load_mode = workload.get("load_mode", "closed_loop")
                concurrency = (
                    int(workload["concurrency"])
                    if load_mode == "closed_loop"
                    else int(workload["max_in_flight"])
                )
                workload_results: list[dict[str, Any]] = []
                workload_started = time.perf_counter()

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    if load_mode == "open_loop":
                        for index in range(request_count):
                            scheduled_offset_s = scheduled_offset_for_request(workload, index)
                            sleep_s = workload_started + scheduled_offset_s - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            futures.append(
                                executor.submit(
                                    post_streaming_chat,
                                    endpoint,
                                    api_key,
                                    model,
                                    workload,
                                    index,
                                    workload_started,
                                    scheduled_offset_s,
                                )
                            )
                    else:
                        futures = [
                            executor.submit(
                                post_streaming_chat,
                                endpoint,
                                api_key,
                                model,
                                workload,
                                index,
                                workload_started,
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
                                "load_mode": load_mode,
                                "arrival_pattern": (
                                    workload.get("arrival_pattern", "steady")
                                    if load_mode == "open_loop"
                                    else "closed_loop"
                                ),
                                "concurrency": (
                                    int(workload["concurrency"])
                                    if load_mode == "closed_loop"
                                    else None
                                ),
                                "target_request_rate_rps": (
                                    float(workload["arrival_rate_rps"])
                                    if load_mode == "open_loop"
                                    else None
                                ),
                                "max_in_flight": concurrency,
                                "requested_max_tokens": int(workload["max_tokens"]),
                                "prompt_tokens_target": (
                                    int(workload["prompt_tokens"])
                                    if "prompt_tokens" in workload
                                    else None
                                ),
                                "prompt_words": (
                                    int(workload["prompt_words"])
                                    if "prompt_words" in workload
                                    else None
                                ),
                            }
                        )
                        workload_results.append(result)
                        request_log.write(json.dumps(result) + "\n")
                        request_log.flush()

                wall_time_s = time.perf_counter() - workload_started
                gpu_summary = load_gpu_summary(gpu_path, workload_name)
                summary_rows.append(
                    summarize_workload(
                        workload_name,
                        workload,
                        workload_results,
                        wall_time_s,
                        gpu_summary,
                        gpu_hourly_cost_usd,
                    )
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
    parser.add_argument(
        "--gpu-hourly-cost-usd",
        type=float,
        help="Optional effective GPU hourly cost used for cost-per-token estimates.",
    )
    return parser.parse_args()


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
