#!/usr/bin/env python3
"""Unit tests for replica selection and controlled fault injection."""

from __future__ import annotations

import http.server
import threading
import unittest
import urllib.request
from unittest.mock import patch

from routing.router import RouterState, WorkerState, handler_for
from routing.slow_worker_proxy import FaultState


def router(policy: str = "round_robin") -> RouterState:
    return RouterState(
        [
            WorkerState("a", "http://a"),
            WorkerState("b", "http://b"),
        ],
        policy,
        failure_threshold=2,
        circuit_open_seconds=10,
        ewma_alpha=0.5,
    )


class RouterPolicyTests(unittest.TestCase):
    def test_round_robin_alternates_workers(self) -> None:
        state = router()
        selected = []
        for _ in range(4):
            worker = state.choose_worker(set())
            assert worker is not None
            selected.append(worker.name)
            state.complete(worker, True, 0.1)
        self.assertEqual(selected, ["a", "b", "a", "b"])

    def test_least_inflight_avoids_busy_worker(self) -> None:
        state = router("least_inflight")
        first = state.choose_worker(set())
        second = state.choose_worker(set())
        assert first is not None and second is not None
        self.assertNotEqual(first.name, second.name)

    def test_latency_aware_prefers_measured_faster_worker(self) -> None:
        state = router("latency_aware")
        a = state.choose_worker(set())
        assert a is not None
        state.complete(a, True, 0.1)
        b = state.choose_worker(set())
        assert b is not None
        state.complete(b, True, 1.0)
        selected = state.choose_worker(set())
        assert selected is not None
        self.assertEqual(selected.name, "a")

    def test_repeated_failures_open_circuit(self) -> None:
        state = router()
        with patch("routing.router.time.monotonic", return_value=100.0):
            for _ in range(2):
                worker = state.choose_worker({"b"})
                assert worker is not None
                state.complete(worker, False, 0.1, "injected")
            selected = state.choose_worker({"b"})
            snapshot = state.snapshot()
        self.assertIsNone(selected)
        worker_a = next(
            worker for worker in snapshot["workers"] if worker["name"] == "a"
        )
        self.assertTrue(worker_a["circuit_open"])

    def test_incomplete_worker_metrics_do_not_publish_zero_queue(self) -> None:
        state = router()
        state.update_upstream_metrics(
            "a",
            running=1,
            waiting=None,
            metrics_up=True,
            metrics_complete=False,
            error="waiting metric missing",
        )
        state.update_upstream_metrics(
            "b",
            running=0,
            waiting=0,
            metrics_up=True,
            metrics_complete=True,
            error=None,
        )
        metrics = state.prometheus()
        self.assertIn("llm_router_upstream_metrics_complete 0", metrics)
        self.assertNotIn("vllm:num_requests_running", metrics)
        self.assertNotIn("vllm:num_requests_waiting", metrics)
        self.assertNotIn(
            'llm_router_worker_upstream_waiting{worker="a"}', metrics
        )
        self.assertIn(
            'llm_router_worker_upstream_waiting{worker="b"} 0', metrics
        )


class RouterStreamingTests(unittest.TestCase):
    def test_retry_records_final_worker_attempt_and_history(self) -> None:
        class FailingUpstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                body = b'{"error":"injected"}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class SuccessfulUpstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                body = b'data: {"token":"ok"}\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        servers = [
            http.server.ThreadingHTTPServer(("127.0.0.1", 0), FailingUpstream),
            http.server.ThreadingHTTPServer(("127.0.0.1", 0), SuccessfulUpstream),
        ]
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in servers
        ]
        for thread in threads:
            thread.start()
        state = RouterState(
            [
                WorkerState("a", f"http://127.0.0.1:{servers[0].server_port}"),
                WorkerState("b", f"http://127.0.0.1:{servers[1].server_port}"),
            ],
            "round_robin",
            failure_threshold=2,
            circuit_open_seconds=10,
            ewma_alpha=0.5,
        )
        router_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_for(state, timeout_s=5, max_retries=1)
        )
        router_thread = threading.Thread(
            target=router_server.serve_forever, daemon=True
        )
        router_thread.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{router_server.server_port}/v1/chat/completions",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read()
                headers = response.headers
            self.assertIn(b"[DONE]", body)
            self.assertEqual(headers.get("X-Router-Worker"), "b")
            self.assertEqual(headers.get("X-Router-Attempt"), "2")
            self.assertEqual(headers.get("X-Router-Attempt-History"), "a,b")
            snapshot = state.snapshot()
            self.assertEqual(snapshot["requests"], 1)
            self.assertEqual(snapshot["retries"], 1)
            self.assertEqual(snapshot["workers"][0]["failures"], 1)
            self.assertEqual(snapshot["workers"][1]["successes"], 1)
        finally:
            router_server.shutdown()
            router_server.server_close()
            router_thread.join(timeout=2)
            for server, thread in zip(servers, threads):
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_router_forwards_first_sse_event_before_upstream_finishes(self) -> None:
        first_event_sent = threading.Event()
        release_second_event = threading.Event()
        first_event_received = threading.Event()
        received_lines: list[bytes] = []
        received_headers: dict[str, str] = {}
        client_errors: list[BaseException] = []

        class StreamingUpstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                if self.path != "/metrics":
                    self.send_error(404)
                    return
                body = (
                    "vllm:num_requests_running 2\n"
                    "vllm:num_requests_waiting 3\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
                content_length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(content_length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"token":"first"}\n\n')
                self.wfile.flush()
                first_event_sent.set()
                release_second_event.wait(timeout=2)
                self.wfile.write(b'data: {"token":"second"}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        upstream_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), StreamingUpstream
        )
        upstream_thread = threading.Thread(
            target=upstream_server.serve_forever, daemon=True
        )
        upstream_thread.start()

        upstream_url = f"http://127.0.0.1:{upstream_server.server_port}"
        state = RouterState(
            [WorkerState("upstream", upstream_url)],
            "round_robin",
            failure_threshold=2,
            circuit_open_seconds=10,
            ewma_alpha=0.5,
        )
        router_server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), handler_for(state, timeout_s=5, max_retries=0)
        )
        router_thread = threading.Thread(
            target=router_server.serve_forever, daemon=True
        )
        router_thread.start()

        with urllib.request.urlopen(
            f"http://127.0.0.1:{router_server.server_port}/metrics", timeout=5
        ) as response:
            router_metrics = response.read().decode("utf-8")
        self.assertIn(
            'llm_router_worker_upstream_running{worker="upstream"} 2.0',
            router_metrics,
        )
        self.assertIn(
            'llm_router_worker_upstream_waiting{worker="upstream"} 3.0',
            router_metrics,
        )
        self.assertIn(
            'llm_router_worker_routable{worker="upstream"} 1', router_metrics
        )
        self.assertIn("llm_router_upstream_metrics_complete 1", router_metrics)

        def read_stream() -> None:
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{router_server.server_port}/v1/chat/completions",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    received_headers.update(response.headers.items())
                    received_lines.append(response.readline())
                    first_event_received.set()
                    received_lines.extend(response.readlines())
            except BaseException as exc:  # noqa: BLE001 - surfaced in test thread.
                client_errors.append(exc)
                first_event_received.set()

        client_thread = threading.Thread(target=read_stream, daemon=True)
        client_thread.start()
        try:
            self.assertTrue(first_event_sent.wait(timeout=1))
            self.assertTrue(
                first_event_received.wait(timeout=1),
                "router buffered the first SSE event until the upstream completed",
            )
            self.assertFalse(client_errors)
            self.assertEqual(received_lines[0], b'data: {"token":"first"}\n')
            self.assertEqual(received_headers.get("X-Router-Worker"), "upstream")
            self.assertEqual(received_headers.get("X-Router-Attempt"), "1")
            self.assertEqual(
                received_headers.get("X-Router-Attempt-History"), "upstream"
            )
        finally:
            release_second_event.set()
            client_thread.join(timeout=2)
            router_server.shutdown()
            router_server.server_close()
            router_thread.join(timeout=2)
            upstream_server.shutdown()
            upstream_server.server_close()
            upstream_thread.join(timeout=2)


class FaultProxyTests(unittest.TestCase):
    def test_every_nth_request_fails(self) -> None:
        state = FaultState(fail_every=3)
        outcomes = [state.next_request()[1] for _ in range(6)]
        self.assertEqual(outcomes, [False, False, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
