#!/usr/bin/env python3
"""Inject deterministic delay or failures in front of one vLLM replica."""

from __future__ import annotations

import argparse
import http.server
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any


class FaultState:
    def __init__(self, fail_every: int) -> None:
        self.fail_every = fail_every
        self.lock = threading.Lock()
        self.requests = 0
        self.injected_failures = 0

    def next_request(self) -> tuple[int, bool]:
        with self.lock:
            self.requests += 1
            should_fail = (
                self.fail_every > 0 and self.requests % self.fail_every == 0
            )
            if should_fail:
                self.injected_failures += 1
            return self.requests, should_fail

    def metrics(self) -> str:
        with self.lock:
            return (
                "# TYPE llm_fault_proxy_requests_total counter\n"
                f"llm_fault_proxy_requests_total {self.requests}\n"
                "# TYPE llm_fault_proxy_injected_failures_total counter\n"
                "llm_fault_proxy_injected_failures_total "
                f"{self.injected_failures}\n"
            )


def handler_for(
    upstream: str,
    delay_ms: float,
    failure_status: int,
    timeout_s: float,
    state: FaultState,
) -> type[http.server.BaseHTTPRequestHandler]:
    class FaultProxyHandler(http.server.BaseHTTPRequestHandler):
        server_version = "LLMFaultProxy/1.0"

        def log_message(self, format_string: str, *args: Any) -> None:
            print(
                json.dumps(
                    {
                        "event": "fault_proxy_access",
                        "message": format_string % args,
                    }
                ),
                flush=True,
            )

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = state.metrics().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            request_number, should_fail = state.next_request()
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000)
            if should_fail:
                payload = json.dumps(
                    {
                        "error": "injected worker failure",
                        "request_number": request_number,
                    }
                ).encode("utf-8")
                self.send_response(failure_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Fault-Injected", "true")
                self.end_headers()
                self.wfile.write(payload)
                return

            headers = {
                "Content-Type": self.headers.get(
                    "Content-Type", "application/json"
                ),
                "Accept": self.headers.get("Accept", "*/*"),
            }
            for name in ("Authorization", "X-Request-ID"):
                if self.headers.get(name):
                    headers[name] = self.headers[name]
            request = urllib.request.Request(
                upstream.rstrip("/") + self.path,
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                response = urllib.request.urlopen(request, timeout=timeout_s)
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                self.send_response(exc.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            except Exception as exc:  # noqa: BLE001 - proxy boundary.
                payload = json.dumps(
                    {"error": "upstream unavailable", "detail": str(exc)}
                ).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            try:
                self.send_response(response.status)
                content_type = response.headers.get(
                    "Content-Type", "application/octet-stream"
                )
                self.send_header("Content-Type", content_type)
                self.send_header("X-Fault-Proxy-Delay-Ms", str(delay_ms))
                self.end_headers()
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                response.close()

    return FaultProxyHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument(
        "--fail-every",
        type=int,
        default=0,
        help="Return an injected failure for every Nth request; zero disables.",
    )
    parser.add_argument("--failure-status", type=int, default=503)
    parser.add_argument("--upstream-timeout", type=float, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delay_ms < 0:
        raise ValueError("delay-ms must be >= 0")
    if args.fail_every < 0:
        raise ValueError("fail-every must be >= 0")
    if not 500 <= args.failure_status <= 599:
        raise ValueError("failure-status must be between 500 and 599")
    if args.upstream_timeout <= 0:
        raise ValueError("upstream-timeout must be > 0")
    state = FaultState(args.fail_every)
    server = http.server.ThreadingHTTPServer(
        (args.host, args.port),
        handler_for(
            args.upstream,
            args.delay_ms,
            args.failure_status,
            args.upstream_timeout,
            state,
        ),
    )
    print(
        json.dumps(
            {
                "event": "fault_proxy_started",
                "upstream": args.upstream,
                "host": args.host,
                "port": args.port,
                "delay_ms": args.delay_ms,
                "fail_every": args.fail_every,
                "failure_status": args.failure_status,
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
