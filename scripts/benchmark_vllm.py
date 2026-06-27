#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI-compatible chat endpoint with streaming TTFT capture."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.server
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


VLLM_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_by_source_total",
    # Compatibility with older vLLM metric names and saved benchmark samples.
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:prompt_tokens_cached",
    "vllm:num_preemptions",
}
VLLM_LOCAL_CACHE_HIT_TOKENS = (
    'vllm:prompt_tokens_by_source_total{source="local_cache_hit"}'
)

SERVER_SETTING_ENV_KEYS = [
    "ENABLE_PREFIX_CACHING",
    "ENABLE_CHUNKED_PREFILL",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_TOKENS",
    "MAX_NUM_PARTIAL_PREFILLS",
    "MAX_LONG_PARTIAL_PREFILLS",
    "LONG_PREFILL_TOKEN_THRESHOLD",
    "GPU_MEMORY_UTILIZATION",
    "KV_CACHE_DTYPE",
    "QUANTIZATION",
    "TENSOR_PARALLEL_SIZE",
    "CPU_OFFLOAD_GB",
]


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


def prompt_token_target_for_request(
    workload: dict[str, Any], request_index: int
) -> int | None:
    pattern = workload.get("prompt_tokens_pattern")
    if pattern:
        return int(pattern[request_index % len(pattern)])
    if workload.get("prompt_tokens") is not None:
        return int(workload["prompt_tokens"])
    return None


def text_with_exact_length(seed: str, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    repetitions = math.ceil(target_chars / len(seed))
    return (seed * repetitions)[:target_chars]


def generated_prefix_cache_prompt(
    token_count: int,
    shared_prefix_tokens: int,
    request_index: int,
    prefix_mode: str,
    response_instruction: str,
) -> str:
    """Build equal-sized prompts whose long prefix is either shared or request-unique."""
    target_chars = token_count * 4
    shared_chars = min(shared_prefix_tokens * 4, target_chars)
    if prefix_mode == "shared":
        prefix_seed = (
            "Shared operating manual: capacity latency throughput queueing prefill "
            "decode memory batching telemetry cache. "
        )
    else:
        prefix_seed = (
            f"Unique request {request_index}: capacity latency throughput queueing "
            "prefill decode memory batching telemetry cache. "
        )
    prefix = text_with_exact_length(prefix_seed, shared_chars)
    suffix_seed = (
        f"\nRequest id: {request_index}\nQuestion: {response_instruction}\n"
        "Return a deterministic operational analysis. "
    )
    suffix = text_with_exact_length(suffix_seed, target_chars - len(prefix))
    return prefix + suffix


def prompt_for_workload(
    workload: dict[str, Any],
    request_index: int,
    prompt_tokens_override: int | None = None,
) -> str:
    if workload.get("prompt"):
        return str(workload["prompt"])
    response_instruction = str(
        workload.get("response_instruction", DEFAULT_RESPONSE_INSTRUCTION)
    )
    prompt_tokens = (
        prompt_tokens_override
        if prompt_tokens_override is not None
        else prompt_token_target_for_request(workload, request_index)
    )
    prefix_mode = workload.get("prefix_mode")
    if prompt_tokens is not None and prefix_mode in {"shared", "unique"}:
        return generated_prefix_cache_prompt(
            prompt_tokens,
            int(workload.get("shared_prefix_tokens", 0)),
            request_index,
            str(prefix_mode),
            response_instruction,
        )
    if prompt_tokens is not None:
        return generated_prompt_for_token_target(
            prompt_tokens,
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


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric_with_labels, raw_value = parts
        metric_name = metric_with_labels.split("{", 1)[0]
        if metric_name not in VLLM_METRICS:
            continue
        if metric_name == "vllm:prompt_tokens_by_source_total":
            if 'source="local_cache_hit"' not in metric_with_labels:
                continue
            metric_name = VLLM_LOCAL_CACHE_HIT_TOKENS
        try:
            metrics[metric_name] += float(raw_value)
        except ValueError:
            continue
    return dict(metrics)


class VllmMetricsSampler:
    def __init__(
        self,
        metrics_url: str,
        output_path: Path,
        run_id: str,
        interval_seconds: float,
    ) -> None:
        self.metrics_url = metrics_url
        self.output_path = output_path
        self.run_id = run_id
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.workload_name = "unassigned"
        self.workload_lock = threading.Lock()
        self.sample_lock = threading.Lock()

    def set_workload(self, workload_name: str) -> None:
        with self.workload_lock:
            self.workload_name = workload_name

    def start(self) -> None:
        self.output_path.write_text("", encoding="utf-8")
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

    def sample_now(self) -> None:
        self._sample_once()

    def _sample_once(self) -> None:
        with self.sample_lock:
            self._sample_once_locked()

    def _sample_once_locked(self) -> None:
        collected_at = utc_now()
        with self.workload_lock:
            workload_name = self.workload_name
        record: dict[str, Any] = {
            "collected_at": collected_at,
            "collected_at_epoch_s": time.time(),
            "run_id": self.run_id,
            "workload": workload_name,
        }
        try:
            with urllib.request.urlopen(self.metrics_url, timeout=5) as response:
                body = response.read().decode("utf-8", "replace")
            record["metrics"] = parse_prometheus_metrics(body)
        except Exception as exc:  # noqa: BLE001 - sampling must not fail the benchmark.
            record["error"] = str(exc)[:500]
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class BenchmarkMetricsState:
    def __init__(self, run_id: str, model: str) -> None:
        self.run_id = run_id
        self.model = model
        self.lock = threading.Lock()
        self.submitted: dict[str, int] = defaultdict(int)
        self.inflight: dict[str, int] = defaultdict(int)
        self.completed: dict[tuple[str, str], int] = defaultdict(int)
        self.errors: dict[tuple[str, str], int] = defaultdict(int)

    def mark_submitted(self, workload: str) -> None:
        with self.lock:
            self.submitted[workload] += 1
            self.inflight[workload] += 1

    def observe(self, workload: str, result: dict[str, Any]) -> None:
        outcome = "success" if result.get("success") else "error"
        error_type = str(result.get("error_type") or "none")
        with self.lock:
            self.inflight[workload] = max(0, self.inflight[workload] - 1)
            self.completed[(workload, outcome)] += 1
            if outcome == "error":
                self.errors[(workload, error_type)] += 1

    def render(self) -> str:
        run_id = prometheus_escape(self.run_id)
        model = prometheus_escape(self.model)
        lines = [
            "# HELP llm_benchmark_run_info Active benchmark run metadata.",
            "# TYPE llm_benchmark_run_info gauge",
            f'llm_benchmark_run_info{{run_id="{run_id}",model="{model}"}} 1',
            "# HELP llm_benchmark_requests_submitted_total Submitted benchmark requests.",
            "# TYPE llm_benchmark_requests_submitted_total counter",
            "# HELP llm_benchmark_inflight_requests Submitted requests not yet completed.",
            "# TYPE llm_benchmark_inflight_requests gauge",
            "# HELP llm_benchmark_requests_completed_total Completed benchmark requests.",
            "# TYPE llm_benchmark_requests_completed_total counter",
            "# HELP llm_benchmark_errors_total Benchmark errors by classification.",
            "# TYPE llm_benchmark_errors_total counter",
        ]
        with self.lock:
            workloads = sorted(set(self.submitted) | set(self.inflight))
            for workload in workloads:
                escaped = prometheus_escape(workload)
                labels = f'run_id="{run_id}",workload="{escaped}"'
                lines.append(
                    f"llm_benchmark_requests_submitted_total{{{labels}}} "
                    f"{self.submitted[workload]}"
                )
                lines.append(
                    f"llm_benchmark_inflight_requests{{{labels}}} {self.inflight[workload]}"
                )
            for (workload, outcome), count in sorted(self.completed.items()):
                escaped = prometheus_escape(workload)
                lines.append(
                    "llm_benchmark_requests_completed_total{"
                    f'run_id="{run_id}",workload="{escaped}",outcome="{outcome}"'
                    f"}} {count}"
                )
            for (workload, error_type), count in sorted(self.errors.items()):
                escaped = prometheus_escape(workload)
                error = prometheus_escape(error_type)
                lines.append(
                    "llm_benchmark_errors_total{"
                    f'run_id="{run_id}",workload="{escaped}",error_type="{error}"'
                    f"}} {count}"
                )
        lines.append("")
        return "\n".join(lines)


class BenchmarkMetricsServer:
    def __init__(self, state: BenchmarkMetricsState, host: str, port: int) -> None:
        handler = self._handler_for(state)
        self.server = http.server.ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def _handler_for(state: BenchmarkMetricsState) -> type[http.server.BaseHTTPRequestHandler]:
        class MetricsHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                body = state.render().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        return MetricsHandler

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def post_streaming_chat(
    endpoint: str,
    api_key: str,
    model: str,
    workload: dict[str, Any],
    request_index: int,
    workload_started_perf: float | None = None,
    scheduled_offset_s: float | None = None,
) -> dict[str, Any]:
    original_prompt_target = prompt_token_target_for_request(workload, request_index)
    effective_prompt_target = original_prompt_target
    admission_action = "none"
    admission_control = workload.get("admission_control") or {}
    max_admitted_prompt_tokens = (
        int(admission_control["max_prompt_tokens"])
        if admission_control.get("max_prompt_tokens") is not None
        else None
    )

    def scheduling_fields(
        request_started_perf: float, ended_perf: float, latency_origin: str
    ) -> dict[str, Any]:
        request_start_offset_s = (
            request_started_perf - workload_started_perf
            if workload_started_perf is not None
            else None
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
            "latency_origin": latency_origin,
            "scheduled_offset_s": scheduled_offset_s,
            "request_start_offset_s": request_start_offset_s,
            "scheduler_delay_s": scheduler_delay_s,
            "scheduled_latency_s": scheduled_latency_s,
        }

    if (
        original_prompt_target is not None
        and max_admitted_prompt_tokens is not None
        and original_prompt_target > max_admitted_prompt_tokens
    ):
        admission_action = str(admission_control.get("action", "reject"))
        if admission_action == "truncate":
            effective_prompt_target = max_admitted_prompt_tokens
        else:
            decided_perf = time.perf_counter()
            return {
                "request_index": request_index,
                "started_at": utc_now(),
                "ended_at": utc_now(),
                "success": False,
                "status_code": None,
                "error_type": "admission_rejected",
                "error_message": (
                    f"prompt target {original_prompt_target} exceeds admission limit "
                    f"{max_admitted_prompt_tokens}"
                ),
                "latency_s": 0.0,
                **scheduling_fields(
                    decided_perf, decided_perf, "client_admission_decision"
                ),
                "timeout": False,
                "oom": False,
                "rejected": True,
                "admission_action": "reject",
                "original_prompt_tokens_target": original_prompt_target,
                "effective_prompt_tokens_target": None,
            }

    prompt = prompt_for_workload(
        workload,
        request_index,
        prompt_tokens_override=effective_prompt_target,
    )
    admission_fields = {
        "admission_action": admission_action,
        "original_prompt_tokens_target": original_prompt_target,
        "effective_prompt_tokens_target": effective_prompt_target,
    }
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
            **scheduling_fields(started_perf, ended_perf, "request_send"),
            **admission_fields,
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
            **scheduling_fields(started_perf, ended_perf, "request_send"),
            **admission_fields,
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
            **scheduling_fields(started_perf, ended_perf, "request_send"),
            **admission_fields,
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
        **scheduling_fields(started_perf, ended_perf, "request_send"),
        **admission_fields,
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


def load_vllm_metrics_summary(path: Path, workload_name: str) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "vllm_requests_running_max": None,
            "vllm_requests_waiting_max": None,
            "vllm_kv_cache_usage_pct_max": None,
            "vllm_prefix_cache_hit_rate": None,
            "vllm_prefix_cache_hit_tokens": None,
            "vllm_prefix_cache_query_tokens": None,
            "vllm_prompt_tokens_cached": None,
            "vllm_preemptions": None,
            "vllm_metrics_sample_errors": 0,
        }
    snapshots: list[dict[str, Any]] = []
    sample_errors = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            sample_errors += 1
            continue
        if record.get("workload") != workload_name:
            continue
        if record.get("error"):
            sample_errors += 1
        elif record.get("metrics"):
            snapshots.append(record)

    def maximum(metric: str, scale: float = 1.0) -> float | None:
        values = [
            safe_float(record.get("metrics", {}).get(metric))
            for record in snapshots
        ]
        present = [value * scale for value in values if value is not None]
        return max(present) if present else None

    def counter_delta(metric: str) -> float | None:
        values = [
            safe_float(record.get("metrics", {}).get(metric))
            for record in snapshots
        ]
        present = [value for value in values if value is not None]
        if not present:
            return None
        if len(present) == 1:
            return 0.0
        return max(0.0, present[-1] - present[0])

    def counter_delta_with_aliases(*metrics: str) -> float | None:
        for metric in metrics:
            value = counter_delta(metric)
            if value is not None:
                return value
        return None

    prefix_hits = counter_delta_with_aliases(
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_hits",
    )
    prefix_queries = counter_delta_with_aliases(
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_queries",
    )
    prompt_tokens_cached = counter_delta_with_aliases(
        "vllm:prompt_tokens_cached_total",
        VLLM_LOCAL_CACHE_HIT_TOKENS,
        "vllm:prompt_tokens_cached",
    )

    return {
        "vllm_requests_running_max": maximum("vllm:num_requests_running"),
        "vllm_requests_waiting_max": maximum("vllm:num_requests_waiting"),
        "vllm_kv_cache_usage_pct_max": maximum("vllm:kv_cache_usage_perc", 100),
        "vllm_prefix_cache_hit_rate": (
            prefix_hits / prefix_queries
            if prefix_hits is not None and prefix_queries not in (None, 0)
            else None
        ),
        "vllm_prefix_cache_hit_tokens": prefix_hits,
        "vllm_prefix_cache_query_tokens": prefix_queries,
        "vllm_prompt_tokens_cached": prompt_tokens_cached,
        "vllm_preemptions": counter_delta("vllm:num_preemptions"),
        "vllm_metrics_sample_errors": sample_errors,
    }


def summarize_workload(
    workload_name: str,
    workload: dict[str, Any],
    results: list[dict[str, Any]],
    wall_time_s: float,
    gpu_summary: dict[str, float | None],
    vllm_metrics_summary: dict[str, float | int | None],
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
    configured_prompt_targets = [
        int(value) for value in (workload.get("prompt_tokens_pattern") or [])
    ]
    if not configured_prompt_targets and workload.get("prompt_tokens") is not None:
        configured_prompt_targets = [int(workload["prompt_tokens"])]
    admission_control = workload.get("admission_control") or {}

    return {
        "workload": workload_name,
        "analysis_group": workload.get("analysis_group"),
        "load_mode": load_mode,
        "arrival_pattern": (
            workload.get("arrival_pattern", "steady")
            if load_mode == "open_loop"
            else "closed_loop"
        ),
        "burst_size": int(workload["burst_size"]) if "burst_size" in workload else None,
        "concurrency": int(workload["concurrency"]) if "concurrency" in workload else None,
        "prompt_tokens_target": (
            configured_prompt_targets[0]
            if len(set(configured_prompt_targets)) == 1
            else None
        ),
        "prompt_tokens_target_min": (
            min(configured_prompt_targets) if configured_prompt_targets else None
        ),
        "prompt_tokens_target_max": (
            max(configured_prompt_targets) if configured_prompt_targets else None
        ),
        "prefix_mode": workload.get("prefix_mode"),
        "shared_prefix_tokens": workload.get("shared_prefix_tokens"),
        "admission_control_action": admission_control.get("action"),
        "admission_max_prompt_tokens": admission_control.get("max_prompt_tokens"),
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
        "admission_rejected_count": sum(
            1 for row in results if row.get("error_type") == "admission_rejected"
        ),
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
        "vllm_requests_running_max": vllm_metrics_summary.get(
            "vllm_requests_running_max"
        ),
        "vllm_requests_waiting_max": vllm_metrics_summary.get(
            "vllm_requests_waiting_max"
        ),
        "vllm_kv_cache_usage_pct_max": vllm_metrics_summary.get(
            "vllm_kv_cache_usage_pct_max"
        ),
        "vllm_prefix_cache_hit_rate": vllm_metrics_summary.get(
            "vllm_prefix_cache_hit_rate"
        ),
        "vllm_prefix_cache_hit_tokens": vllm_metrics_summary.get(
            "vllm_prefix_cache_hit_tokens"
        ),
        "vllm_prefix_cache_query_tokens": vllm_metrics_summary.get(
            "vllm_prefix_cache_query_tokens"
        ),
        "vllm_prompt_tokens_cached": vllm_metrics_summary.get(
            "vllm_prompt_tokens_cached"
        ),
        "vllm_preemptions": vllm_metrics_summary.get("vllm_preemptions"),
        "vllm_metrics_sample_errors": vllm_metrics_summary.get(
            "vllm_metrics_sample_errors"
        ),
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
        "analysis_group",
        "load_mode",
        "arrival_pattern",
        "burst_size",
        "concurrency",
        "prompt_tokens_target",
        "prompt_tokens_target_min",
        "prompt_tokens_target_max",
        "prefix_mode",
        "shared_prefix_tokens",
        "admission_control_action",
        "admission_max_prompt_tokens",
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
        "admission_rejected_count",
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
        "vllm_requests_running_max",
        "vllm_requests_waiting_max",
        "vllm_kv_cache_usage_pct_max",
        "vllm_prefix_cache_hit_rate",
        "vllm_prefix_cache_hit_tokens",
        "vllm_prefix_cache_query_tokens",
        "vllm_prompt_tokens_cached",
        "vllm_preemptions",
        "vllm_metrics_sample_errors",
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
    max_model_len = config.get("max_model_len")
    if max_model_len is not None:
        max_model_len = int(max_model_len)
        if max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
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
        prompt_pattern = run.get("prompt_tokens_pattern")
        if prompt_pattern is not None:
            if not isinstance(prompt_pattern, list) or not prompt_pattern:
                raise ValueError(
                    f"run {run['name']} prompt_tokens_pattern must be a non-empty list"
                )
            if "prompt_tokens" in run:
                raise ValueError(
                    f"run {run['name']} cannot define both prompt_tokens and "
                    "prompt_tokens_pattern"
                )
            if any(int(value) < 1 for value in prompt_pattern):
                raise ValueError(
                    f"run {run['name']} prompt_tokens_pattern values must be >= 1"
                )
        if "prompt_tokens" in run and int(run["prompt_tokens"]) < 1:
            raise ValueError(f"run {run['name']} must have prompt_tokens >= 1")
        if "output_tokens_target" in run and int(run["output_tokens_target"]) < 1:
            raise ValueError(f"run {run['name']} must have output_tokens_target >= 1")
        configured_prompt_targets = [int(value) for value in (prompt_pattern or [])]
        if not configured_prompt_targets and "prompt_tokens" in run:
            configured_prompt_targets = [int(run["prompt_tokens"])]
        if max_model_len is not None and configured_prompt_targets:
            target_context_tokens = max(configured_prompt_targets) + int(run["max_tokens"])
            if target_context_tokens >= max_model_len:
                raise ValueError(
                    f"run {run['name']} targets {target_context_tokens} prompt plus "
                    f"output tokens; this must be less than max_model_len "
                    f"{max_model_len}"
                )
        prefix_mode = run.get("prefix_mode")
        if prefix_mode is not None and prefix_mode not in {"shared", "unique"}:
            raise ValueError(
                f"run {run['name']} prefix_mode must be 'shared' or 'unique'"
            )
        if prefix_mode is not None:
            shared_prefix_tokens = int(run.get("shared_prefix_tokens", 0))
            if shared_prefix_tokens < 1:
                raise ValueError(
                    f"run {run['name']} must define shared_prefix_tokens >= 1"
                )
            if configured_prompt_targets and shared_prefix_tokens >= min(
                configured_prompt_targets
            ):
                raise ValueError(
                    f"run {run['name']} shared_prefix_tokens must be smaller than "
                    "every prompt target"
                )
        admission_control = run.get("admission_control")
        if admission_control is not None:
            if not isinstance(admission_control, dict):
                raise ValueError(f"run {run['name']} admission_control must be an object")
            action = admission_control.get("action", "reject")
            if action not in {"reject", "truncate"}:
                raise ValueError(
                    f"run {run['name']} admission action must be reject or truncate"
                )
            max_admitted_prompt_tokens = safe_float(
                admission_control.get("max_prompt_tokens")
            )
            if (
                max_admitted_prompt_tokens is None
                or max_admitted_prompt_tokens < 1
            ):
                raise ValueError(
                    f"run {run['name']} admission max_prompt_tokens must be >= 1"
                )
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
        "max_model_len": config.get("max_model_len"),
        "runs": [
            {
                "name": run["name"],
                "analysis_group": run.get("analysis_group"),
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
                "prompt_tokens": prompt_token_target_for_request(run, 0),
                "prompt_tokens_pattern": run.get("prompt_tokens_pattern"),
                "prompt_words": int(run["prompt_words"]) if "prompt_words" in run else None,
                "max_tokens": int(run["max_tokens"]),
                "target_context_tokens": (
                    max(
                        int(value)
                        for value in (
                            run.get("prompt_tokens_pattern")
                            or [run.get("prompt_tokens")]
                        )
                        if value is not None
                    )
                    + int(run["max_tokens"])
                    if run.get("prompt_tokens_pattern") or "prompt_tokens" in run
                    else None
                ),
                "context_headroom_tokens": (
                    int(config["max_model_len"])
                    - max(
                        int(value)
                        for value in (
                            run.get("prompt_tokens_pattern")
                            or [run.get("prompt_tokens")]
                        )
                        if value is not None
                    )
                    - int(run["max_tokens"])
                    if "max_model_len" in config
                    and (run.get("prompt_tokens_pattern") or "prompt_tokens" in run)
                    else None
                ),
                "prefix_mode": run.get("prefix_mode"),
                "shared_prefix_tokens": run.get("shared_prefix_tokens"),
                "admission_control": run.get("admission_control"),
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
    if args.gpu_sample_interval <= 0:
        raise ValueError("gpu-sample-interval must be > 0")
    if getattr(args, "vllm_metrics_interval", 2.0) <= 0:
        raise ValueError("vllm-metrics-interval must be > 0")
    metrics_export_port = getattr(args, "metrics_export_port", None)
    if metrics_export_port is not None and not 1 <= metrics_export_port <= 65535:
        raise ValueError("metrics-export-port must be between 1 and 65535")

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
    vllm_metrics_path = run_dir / "vllm_metrics.jsonl"
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
                "max_model_len": config.get("max_model_len"),
                "server_config_label": (
                    getattr(args, "server_config_label", None)
                    or os.environ.get("VLLM_SERVER_CONFIG_LABEL")
                ),
                "server_settings": {
                    key: os.environ[key]
                    for key in SERVER_SETTING_ENV_KEYS
                    if key in os.environ
                },
                "gpu_hourly_cost_usd": gpu_hourly_cost_usd,
                "vllm_metrics_sampling_enabled": not getattr(
                    args, "disable_vllm_metrics_sampling", False
                ),
                "benchmark_metrics_export_port": getattr(
                    args, "metrics_export_port", None
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gpu_sampler = GpuSampler(gpu_path, args.gpu_sample_interval)
    vllm_sampler = (
        None
        if getattr(args, "disable_vllm_metrics_sampling", False)
        else VllmMetricsSampler(
            base_url.rstrip("/") + "/metrics",
            vllm_metrics_path,
            run_id,
            getattr(args, "vllm_metrics_interval", 2.0),
        )
    )
    metrics_state = (
        BenchmarkMetricsState(run_id, model)
        if getattr(args, "metrics_export_port", None) is not None
        else None
    )
    metrics_server = (
        BenchmarkMetricsServer(
            metrics_state,
            getattr(args, "metrics_export_host", "0.0.0.0"),
            int(args.metrics_export_port),
        )
        if metrics_state is not None
        else None
    )
    gpu_sampler.start()
    if vllm_sampler is not None:
        vllm_sampler.start()
    if metrics_server is not None:
        metrics_server.start()
    summary_rows: list[dict[str, Any]] = []

    try:
        with request_log_path.open("w", encoding="utf-8") as request_log:
            for workload in config["runs"]:
                workload_name = workload["name"]
                gpu_sampler.set_workload(workload_name)
                if vllm_sampler is not None:
                    vllm_sampler.set_workload(workload_name)
                    vllm_sampler.sample_now()
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

                    def register_future(future: Any) -> Any:
                        if metrics_state is not None:
                            metrics_state.mark_submitted(workload_name)

                            def observe_completed(completed: Any) -> None:
                                try:
                                    metrics_state.observe(workload_name, completed.result())
                                except Exception as exc:  # noqa: BLE001
                                    metrics_state.observe(
                                        workload_name,
                                        {"success": False, "error_type": type(exc).__name__},
                                    )

                            future.add_done_callback(observe_completed)
                        return future

                    if load_mode == "open_loop":
                        for index in range(request_count):
                            scheduled_offset_s = scheduled_offset_for_request(workload, index)
                            sleep_s = workload_started + scheduled_offset_s - time.perf_counter()
                            if sleep_s > 0:
                                time.sleep(sleep_s)
                            futures.append(
                                register_future(executor.submit(
                                    post_streaming_chat,
                                    endpoint,
                                    api_key,
                                    model,
                                    workload,
                                    index,
                                    workload_started,
                                    scheduled_offset_s,
                                ))
                            )
                    else:
                        futures = [
                            register_future(executor.submit(
                                post_streaming_chat,
                                endpoint,
                                api_key,
                                model,
                                workload,
                                index,
                                workload_started,
                            ))
                            for index in range(request_count)
                        ]
                    for future in as_completed(futures):
                        result = future.result()
                        completed_request_index = int(result["request_index"])
                        result.update(
                            {
                                "run_id": run_id,
                                "workload": workload_name,
                                "analysis_group": workload.get("analysis_group"),
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
                                    prompt_token_target_for_request(
                                        workload, completed_request_index
                                    )
                                ),
                                "prefix_mode": workload.get("prefix_mode"),
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
                if vllm_sampler is not None:
                    vllm_sampler.sample_now()
                gpu_summary = load_gpu_summary(gpu_path, workload_name)
                vllm_metrics_summary = load_vllm_metrics_summary(
                    vllm_metrics_path, workload_name
                )
                summary_rows.append(
                    summarize_workload(
                        workload_name,
                        workload,
                        workload_results,
                        wall_time_s,
                        gpu_summary,
                        vllm_metrics_summary,
                        gpu_hourly_cost_usd,
                    )
                )
    finally:
        if metrics_server is not None:
            metrics_server.stop()
        if vllm_sampler is not None:
            vllm_sampler.stop()
        gpu_sampler.stop()

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
    parser.add_argument(
        "--server-config-label",
        help="Label recorded in metadata for cross-run serving-knob comparisons.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-sample-interval", type=float, default=2.0)
    parser.add_argument("--vllm-metrics-interval", type=float, default=2.0)
    parser.add_argument("--disable-vllm-metrics-sampling", action="store_true")
    parser.add_argument("--metrics-export-host", default="0.0.0.0")
    parser.add_argument(
        "--metrics-export-port",
        type=int,
        help="Optional port for live benchmark error/timeout/OOM Prometheus metrics.",
    )
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
