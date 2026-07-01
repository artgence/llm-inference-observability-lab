#!/usr/bin/env python3
"""Unit tests for replica selection and controlled fault injection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from routing.router import RouterState, WorkerState
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


class FaultProxyTests(unittest.TestCase):
    def test_every_nth_request_fails(self) -> None:
        state = FaultState(fail_every=3)
        outcomes = [state.next_request()[1] for _ in range(6)]
        self.assertEqual(outcomes, [False, False, True, False, False, True])


if __name__ == "__main__":
    unittest.main()
