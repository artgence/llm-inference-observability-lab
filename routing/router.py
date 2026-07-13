#!/usr/bin/env python3
"""Small streaming router for independent OpenAI-compatible vLLM replicas."""

from __future__ import annotations

import argparse
import http.server
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
WORKER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def prometheus_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@dataclass
class WorkerState:
    name: str
    base_url: str
    inflight: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    circuit_open_until: float = 0.0
    ewma_latency_s: float | None = None
    last_error: str | None = None
    attempt_duration_sum_s: float = 0.0
    attempt_duration_count: int = 0
    cancellations: int = 0
    upstream_running: float | None = None
    upstream_waiting: float | None = None
    upstream_metrics_up: bool | None = None
    upstream_metrics_complete: bool | None = None
    upstream_metrics_error: str | None = None


class RouterState:
    def __init__(
        self,
        workers: list[WorkerState],
        policy: str,
        failure_threshold: int,
        circuit_open_seconds: float,
        ewma_alpha: float,
    ) -> None:
        self.workers = workers
        self.policy = policy
        self.failure_threshold = failure_threshold
        self.circuit_open_seconds = circuit_open_seconds
        self.ewma_alpha = ewma_alpha
        self.lock = threading.Lock()
        self.round_robin_index = 0
        self.requests = 0
        self.retries = 0
        self.no_healthy_worker = 0

    def begin_request(self) -> None:
        with self.lock:
            self.requests += 1

    def mark_retry(self) -> None:
        with self.lock:
            self.retries += 1

    def worker_targets(self) -> list[tuple[str, str]]:
        with self.lock:
            return [(worker.name, worker.base_url) for worker in self.workers]

    def update_upstream_metrics(
        self,
        worker_name: str,
        running: float | None,
        waiting: float | None,
        metrics_up: bool,
        metrics_complete: bool,
        error: str | None,
    ) -> None:
        with self.lock:
            worker = next(
                (candidate for candidate in self.workers if candidate.name == worker_name),
                None,
            )
            if worker is None:
                return
            worker.upstream_running = running
            worker.upstream_waiting = waiting
            worker.upstream_metrics_up = metrics_up
            worker.upstream_metrics_complete = metrics_complete
            worker.upstream_metrics_error = error

    def choose_worker(self, excluded: set[str]) -> WorkerState | None:
        with self.lock:
            now = time.monotonic()
            candidates = [
                worker
                for worker in self.workers
                if worker.name not in excluded and worker.circuit_open_until <= now
            ]
            if not candidates:
                self.no_healthy_worker += 1
                return None
            if self.policy == "round_robin":
                for _ in range(len(self.workers)):
                    worker = self.workers[
                        self.round_robin_index % len(self.workers)
                    ]
                    self.round_robin_index += 1
                    if worker in candidates:
                        selected = worker
                        break
                else:
                    selected = candidates[0]
            elif self.policy == "least_inflight":
                selected = min(
                    candidates,
                    key=lambda worker: (
                        worker.inflight,
                        worker.attempts,
                        worker.name,
                    ),
                )
            else:
                selected = min(
                    candidates,
                    key=lambda worker: (
                        worker.ewma_latency_s is not None,
                        (worker.ewma_latency_s or 1.0) * (worker.inflight + 1),
                        worker.inflight,
                        worker.name,
                    ),
                )
            selected.inflight += 1
            selected.attempts += 1
            return selected

    def complete(
        self,
        worker: WorkerState,
        success: bool,
        latency_s: float,
        error: str | None = None,
        circuit_failure: bool = True,
    ) -> None:
        with self.lock:
            worker.inflight = max(0, worker.inflight - 1)
            worker.attempt_duration_sum_s += latency_s
            worker.attempt_duration_count += 1
            if worker.ewma_latency_s is None:
                worker.ewma_latency_s = latency_s
            else:
                worker.ewma_latency_s = (
                    self.ewma_alpha * latency_s
                    + (1 - self.ewma_alpha) * worker.ewma_latency_s
                )
            if success:
                worker.successes += 1
                worker.consecutive_failures = 0
                worker.last_error = None
                return
            worker.failures += 1
            worker.last_error = error
            if not circuit_failure:
                return
            worker.consecutive_failures += 1
            if worker.consecutive_failures >= self.failure_threshold:
                worker.circuit_open_until = (
                    time.monotonic() + self.circuit_open_seconds
                )

    def cancel(
        self,
        worker: WorkerState,
        latency_s: float,
        error: str,
    ) -> None:
        with self.lock:
            worker.inflight = max(0, worker.inflight - 1)
            worker.attempt_duration_sum_s += latency_s
            worker.attempt_duration_count += 1
            if worker.ewma_latency_s is None:
                worker.ewma_latency_s = latency_s
            else:
                worker.ewma_latency_s = (
                    self.ewma_alpha * latency_s
                    + (1 - self.ewma_alpha) * worker.ewma_latency_s
                )
            worker.cancellations += 1
            worker.last_error = error

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = time.monotonic()
            return {
                "policy": self.policy,
                "requests": self.requests,
                "retries": self.retries,
                "no_healthy_worker": self.no_healthy_worker,
                "workers": [
                    {
                        "name": worker.name,
                        "base_url": worker.base_url,
                        "inflight": worker.inflight,
                        "attempts": worker.attempts,
                        "successes": worker.successes,
                        "failures": worker.failures,
                        "consecutive_failures": worker.consecutive_failures,
                        "circuit_open": worker.circuit_open_until > now,
                        "ewma_latency_s": worker.ewma_latency_s,
                        "last_error": worker.last_error,
                        "attempt_duration_sum_s": worker.attempt_duration_sum_s,
                        "attempt_duration_count": worker.attempt_duration_count,
                        "cancellations": worker.cancellations,
                        "upstream_running": worker.upstream_running,
                        "upstream_waiting": worker.upstream_waiting,
                        "upstream_metrics_up": worker.upstream_metrics_up,
                        "upstream_metrics_complete": (
                            worker.upstream_metrics_complete
                        ),
                        "upstream_metrics_error": worker.upstream_metrics_error,
                        "health_state": (
                            "circuit_open"
                            if worker.circuit_open_until > now
                            else "metrics_unreachable"
                            if worker.upstream_metrics_up is False
                            else "routable_metrics_up"
                            if worker.upstream_metrics_up is True
                            else "unknown"
                        ),
                    }
                    for worker in self.workers
                ],
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        metrics_complete = bool(snapshot["workers"]) and all(
            worker["upstream_metrics_complete"] is True
            and worker["upstream_running"] is not None
            and worker["upstream_waiting"] is not None
            for worker in snapshot["workers"]
        )
        lines = [
            "# TYPE llm_router_requests_total counter",
            f"llm_router_requests_total {snapshot['requests']}",
            "# TYPE llm_router_retries_total counter",
            f"llm_router_retries_total {snapshot['retries']}",
            "# TYPE llm_router_no_healthy_worker_total counter",
            "llm_router_no_healthy_worker_total "
            + str(snapshot["no_healthy_worker"]),
            "# TYPE llm_router_upstream_metrics_complete gauge",
            "llm_router_upstream_metrics_complete "
            + ("1" if metrics_complete else "0"),
        ]
        if metrics_complete:
            lines.extend(
                [
                    "# TYPE vllm:num_requests_running gauge",
                    "vllm:num_requests_running "
                    + str(
                        sum(
                            worker["upstream_running"]
                            for worker in snapshot["workers"]
                        )
                    ),
                    "# TYPE vllm:num_requests_waiting gauge",
                    "vllm:num_requests_waiting "
                    + str(
                        sum(
                            worker["upstream_waiting"]
                            for worker in snapshot["workers"]
                        )
                    ),
                ]
            )
        for worker in snapshot["workers"]:
            label = f'worker="{prometheus_escape(worker["name"])}"'
            worker_lines = [
                f"llm_router_worker_inflight{{{label}}} {worker['inflight']}",
                f"llm_router_worker_attempts_total{{{label}}} "
                f"{worker['attempts']}",
                f"llm_router_worker_successes_total{{{label}}} "
                f"{worker['successes']}",
                f"llm_router_worker_failures_total{{{label}}} "
                f"{worker['failures']}",
                f"llm_router_worker_cancellations_total{{{label}}} "
                f"{worker['cancellations']}",
                f"llm_router_worker_circuit_open{{{label}}} "
                f"{1 if worker['circuit_open'] else 0}",
                f"llm_router_worker_attempt_duration_seconds_sum{{{label}}} "
                f"{worker['attempt_duration_sum_s']}",
                f"llm_router_worker_attempt_duration_seconds_count{{{label}}} "
                f"{worker['attempt_duration_count']}",
                f"llm_router_worker_upstream_metrics_up{{{label}}} "
                f"{1 if worker['upstream_metrics_up'] is True else 0}",
                f"llm_router_worker_upstream_metrics_complete{{{label}}} "
                f"{1 if worker['upstream_metrics_complete'] is True else 0}",
                f"llm_router_worker_routable{{{label}}} "
                f"{0 if worker['circuit_open'] else 1}",
            ]
            if worker["ewma_latency_s"] is not None:
                worker_lines.append(
                    f"llm_router_worker_ewma_latency_seconds{{{label}}} "
                    f"{worker['ewma_latency_s']}"
                )
            if worker["upstream_running"] is not None:
                worker_lines.append(
                    f"llm_router_worker_upstream_running{{{label}}} "
                    f"{worker['upstream_running']}"
                )
            if worker["upstream_waiting"] is not None:
                worker_lines.append(
                    f"llm_router_worker_upstream_waiting{{{label}}} "
                    f"{worker['upstream_waiting']}"
                )
            lines.extend(worker_lines)
        lines.append("")
        return "\n".join(lines)


def prometheus_metric_sum(text: str, *metric_names: str) -> float | None:
    for expected in metric_names:
        values: list[float] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            metric_with_labels, raw_value = parts
            if metric_with_labels.split("{", 1)[0] != expected:
                continue
            try:
                values.append(float(raw_value))
            except ValueError:
                continue
        if values:
            return sum(values)
    return None


def fetch_worker_load(
    worker_name: str,
    base_url: str,
    timeout_s: float,
) -> tuple[
    str,
    float | None,
    float | None,
    bool,
    bool,
    str | None,
]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/metrics",
        headers={"User-Agent": "llm-inference-observability-router/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            text = response.read().decode("utf-8", "replace")
        running = prometheus_metric_sum(
            text, "vllm:num_requests_running", "vllm_requests_running"
        )
        waiting = prometheus_metric_sum(
            text, "vllm:num_requests_waiting", "vllm_requests_waiting"
        )
        complete = running is not None and waiting is not None
        error = None if complete else "worker running/waiting metrics were incomplete"
        return worker_name, running, waiting, True, complete, error
    except Exception as exc:  # noqa: BLE001 - metrics must not break routing.
        return worker_name, None, None, False, False, str(exc)[:200]


def refresh_worker_load(state: RouterState, timeout_s: float = 1.0) -> None:
    targets = state.worker_targets()
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=len(targets)) as executor:
        futures = [
            executor.submit(fetch_worker_load, name, base_url, timeout_s)
            for name, base_url in targets
        ]
        for future in as_completed(futures):
            name, running, waiting, metrics_up, complete, error = future.result()
            state.update_upstream_metrics(
                name,
                running,
                waiting,
                metrics_up,
                complete,
                error,
            )


def handler_for(
    state: RouterState,
    timeout_s: float,
    max_retries: int,
) -> type[http.server.BaseHTTPRequestHandler]:
    class RouterHandler(http.server.BaseHTTPRequestHandler):
        server_version = "LLMReplicaRouter/1.0"

        def log_message(self, format_string: str, *args: Any) -> None:
            print(
                json.dumps(
                    {
                        "event": "router_access",
                        "client": self.client_address[0],
                        "message": format_string % args,
                    }
                ),
                flush=True,
            )

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_attempt(
            self,
            request_id: str,
            worker: WorkerState,
            attempt_number: int,
            outcome: str,
            latency_s: float | None = None,
            error: str | None = None,
        ) -> None:
            print(
                json.dumps(
                    {
                        "event": "router_attempt_" + outcome,
                        "request_id": request_id,
                        "path": self.path,
                        "router_selected_worker": worker.name,
                        "router_attempt_number": attempt_number,
                        "latency_s": latency_s,
                        "error": error,
                    }
                ),
                flush=True,
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            if self.path == "/metrics":
                refresh_worker_load(state)
                body = state.prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header(
                    "Content-Type", "text/plain; version=0.0.4"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/health":
                snapshot = state.snapshot()
                healthy = any(
                    not worker["circuit_open"] for worker in snapshot["workers"]
                )
                self.send_json(200 if healthy else 503, snapshot)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            request_id = self.headers.get("X-Request-ID") or str(uuid.uuid4())
            state.begin_request()
            excluded: set[str] = set()
            last_error = "no worker selected"
            last_status = 503
            last_body = b""
            last_worker: str | None = None
            last_attempt_number: int | None = None
            attempt_history: list[str] = []
            for attempt in range(max_retries + 1):
                worker = state.choose_worker(excluded)
                if worker is None:
                    break
                excluded.add(worker.name)
                attempt_number = attempt + 1
                if attempt_number > 1:
                    state.mark_retry()
                last_worker = worker.name
                last_attempt_number = attempt_number
                attempt_history.append(worker.name)
                print(
                    json.dumps(
                        {
                            "event": "router_attempt_started",
                            "request_id": request_id,
                            "path": self.path,
                            "router_selected_worker": worker.name,
                            "router_attempt_number": attempt_number,
                        }
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                upstream_url = worker.base_url.rstrip("/") + self.path
                headers = {
                    "Content-Type": self.headers.get(
                        "Content-Type", "application/json"
                    ),
                    "Accept": self.headers.get("Accept", "*/*"),
                    "X-Request-ID": request_id,
                }
                authorization = self.headers.get("Authorization")
                if authorization:
                    headers["Authorization"] = authorization
                request = urllib.request.Request(
                    upstream_url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                try:
                    upstream = urllib.request.urlopen(request, timeout=timeout_s)
                except urllib.error.HTTPError as exc:
                    latency_s = time.perf_counter() - started
                    last_status = exc.code
                    last_body = exc.read()
                    last_error = f"HTTP {exc.code}"
                    retryable = exc.code >= 500
                    state.complete(
                        worker,
                        False,
                        latency_s,
                        last_error,
                        circuit_failure=retryable,
                    )
                    self.log_attempt(
                        request_id,
                        worker,
                        attempt_number,
                        "failed",
                        latency_s,
                        last_error,
                    )
                    if retryable and attempt < max_retries:
                        continue
                    break
                except Exception as exc:  # noqa: BLE001 - proxy boundary.
                    latency_s = time.perf_counter() - started
                    last_status = 502
                    last_error = str(exc)
                    last_body = json.dumps(
                        {"error": "upstream unavailable", "detail": last_error}
                    ).encode("utf-8")
                    state.complete(worker, False, latency_s, last_error)
                    self.log_attempt(
                        request_id,
                        worker,
                        attempt_number,
                        "failed",
                        latency_s,
                        last_error,
                    )
                    if attempt < max_retries:
                        continue
                    break

                try:
                    self.send_response(upstream.status)
                    for name, value in upstream.headers.items():
                        if name.lower() not in HOP_BY_HOP_HEADERS:
                            self.send_header(name, value)
                    self.send_header("X-Request-ID", request_id)
                    self.send_header("X-Router-Worker", worker.name)
                    self.send_header("X-Router-Attempt", str(attempt_number))
                    self.send_header(
                        "X-Router-Attempt-History", ",".join(attempt_history)
                    )
                    self.end_headers()
                    # vLLM streams OpenAI responses as newline-delimited SSE.
                    # HTTPResponse.read(n) waits for n bytes or EOF, which collapses
                    # a typical generation into one final client-visible chunk.
                    # Iterating by line preserves each SSE event boundary and lets
                    # the benchmark observe real TTFT and inter-token timing.
                    for chunk in upstream:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    latency_s = time.perf_counter() - started
                    state.complete(worker, True, latency_s)
                    self.log_attempt(
                        request_id,
                        worker,
                        attempt_number,
                        "succeeded",
                        latency_s,
                    )
                    return
                except (BrokenPipeError, ConnectionResetError):
                    latency_s = time.perf_counter() - started
                    state.cancel(
                        worker,
                        latency_s,
                        "client disconnected",
                    )
                    self.log_attempt(
                        request_id,
                        worker,
                        attempt_number,
                        "cancelled",
                        latency_s,
                        "client disconnected",
                    )
                    return
                except Exception as exc:  # noqa: BLE001 - streaming boundary.
                    latency_s = time.perf_counter() - started
                    state.complete(
                        worker,
                        False,
                        latency_s,
                        str(exc),
                    )
                    self.log_attempt(
                        request_id,
                        worker,
                        attempt_number,
                        "failed",
                        latency_s,
                        str(exc),
                    )
                    return
                finally:
                    upstream.close()

            self.send_response(last_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(last_body)))
            self.send_header("X-Request-ID", request_id)
            self.send_header("X-Router-Error", last_error[:200])
            if last_worker is not None:
                self.send_header("X-Router-Worker", last_worker)
            if last_attempt_number is not None:
                self.send_header(
                    "X-Router-Attempt", str(last_attempt_number)
                )
            if attempt_history:
                self.send_header(
                    "X-Router-Attempt-History", ",".join(attempt_history)
                )
            self.end_headers()
            self.wfile.write(last_body)

    return RouterHandler


def parse_worker(value: str) -> WorkerState:
    if "=" not in value:
        raise argparse.ArgumentTypeError("worker must use NAME=BASE_URL")
    name, base_url = value.split("=", 1)
    if not WORKER_NAME_PATTERN.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "worker name must contain only letters, digits, dot, underscore, or dash"
        )
    if not base_url.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("worker must use NAME=http://host:port")
    return WorkerState(name=name, base_url=base_url.rstrip("/"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="append",
        type=parse_worker,
        required=True,
        help="Replica endpoint as NAME=BASE_URL; repeat at least twice.",
    )
    parser.add_argument(
        "--policy",
        choices=["round_robin", "least_inflight", "latency_aware"],
        default="round_robin",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--upstream-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, choices=[0, 1], default=0)
    parser.add_argument("--failure-threshold", type=int, default=2)
    parser.add_argument("--circuit-open-seconds", type=float, default=15.0)
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.worker) < 2:
        raise ValueError("at least two --worker endpoints are required")
    if args.upstream_timeout <= 0:
        raise ValueError("upstream-timeout must be > 0")
    if args.failure_threshold < 1:
        raise ValueError("failure-threshold must be >= 1")
    if args.circuit_open_seconds <= 0:
        raise ValueError("circuit-open-seconds must be > 0")
    if not 0 < args.ewma_alpha <= 1:
        raise ValueError("ewma-alpha must be in (0, 1]")
    state = RouterState(
        args.worker,
        args.policy,
        args.failure_threshold,
        args.circuit_open_seconds,
        args.ewma_alpha,
    )
    server = http.server.ThreadingHTTPServer(
        (args.host, args.port),
        handler_for(state, args.upstream_timeout, args.max_retries),
    )
    print(
        json.dumps(
            {
                "event": "router_started",
                "host": args.host,
                "port": args.port,
                "policy": args.policy,
                "workers": [
                    {"name": worker.name, "base_url": worker.base_url}
                    for worker in args.worker
                ],
                "max_retries": args.max_retries,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
