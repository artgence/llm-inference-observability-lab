#!/usr/bin/env python3
"""Small streaming router for independent OpenAI-compatible vLLM replicas."""

from __future__ import annotations

import argparse
import http.server
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
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
    ) -> None:
        with self.lock:
            worker.inflight = max(0, worker.inflight - 1)
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
            worker.consecutive_failures += 1
            worker.last_error = error
            if worker.consecutive_failures >= self.failure_threshold:
                worker.circuit_open_until = (
                    time.monotonic() + self.circuit_open_seconds
                )

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
                    }
                    for worker in self.workers
                ],
            }

    def prometheus(self) -> str:
        snapshot = self.snapshot()
        lines = [
            "# TYPE vllm:num_requests_running gauge",
            "vllm:num_requests_running "
            + str(sum(worker["inflight"] for worker in snapshot["workers"])),
            "# TYPE vllm:num_requests_waiting gauge",
            "vllm:num_requests_waiting 0",
            "# TYPE llm_router_requests_total counter",
            f"llm_router_requests_total {snapshot['requests']}",
            "# TYPE llm_router_retries_total counter",
            f"llm_router_retries_total {snapshot['retries']}",
            "# TYPE llm_router_no_healthy_worker_total counter",
            "llm_router_no_healthy_worker_total "
            + str(snapshot["no_healthy_worker"]),
        ]
        for worker in snapshot["workers"]:
            label = f'worker="{prometheus_escape(worker["name"])}"'
            lines.extend(
                [
                    f"llm_router_worker_inflight{{{label}}} {worker['inflight']}",
                    f"llm_router_worker_attempts_total{{{label}}} "
                    f"{worker['attempts']}",
                    f"llm_router_worker_successes_total{{{label}}} "
                    f"{worker['successes']}",
                    f"llm_router_worker_failures_total{{{label}}} "
                    f"{worker['failures']}",
                    f"llm_router_worker_circuit_open{{{label}}} "
                    f"{1 if worker['circuit_open'] else 0}",
                    f"llm_router_worker_ewma_latency_seconds{{{label}}} "
                    f"{worker['ewma_latency_s'] or 0}",
                ]
            )
        lines.append("")
        return "\n".join(lines)


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

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            if self.path == "/metrics":
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
            for attempt in range(max_retries + 1):
                worker = state.choose_worker(excluded)
                if worker is None:
                    break
                excluded.add(worker.name)
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
                    state.complete(worker, False, latency_s, last_error)
                    if retryable and attempt < max_retries:
                        state.mark_retry()
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
                    if attempt < max_retries:
                        state.mark_retry()
                        continue
                    break

                try:
                    self.send_response(upstream.status)
                    for name, value in upstream.headers.items():
                        if name.lower() not in HOP_BY_HOP_HEADERS:
                            self.send_header(name, value)
                    self.send_header("X-Request-ID", request_id)
                    self.send_header("X-Router-Worker", worker.name)
                    self.end_headers()
                    while True:
                        chunk = upstream.read(64 * 1024)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    state.complete(
                        worker,
                        True,
                        time.perf_counter() - started,
                    )
                    return
                except (BrokenPipeError, ConnectionResetError):
                    state.complete(
                        worker,
                        True,
                        time.perf_counter() - started,
                        "client disconnected",
                    )
                    return
                except Exception as exc:  # noqa: BLE001 - streaming boundary.
                    state.complete(
                        worker,
                        False,
                        time.perf_counter() - started,
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
            self.end_headers()
            self.wfile.write(last_body)

    return RouterHandler


def parse_worker(value: str) -> WorkerState:
    if "=" not in value:
        raise argparse.ArgumentTypeError("worker must use NAME=BASE_URL")
    name, base_url = value.split("=", 1)
    if not name or not base_url.startswith(("http://", "https://")):
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
