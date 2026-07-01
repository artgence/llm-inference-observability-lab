#!/usr/bin/env python3
"""Regression tests for benchmark controls and vLLM metric windows."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_vllm import (
    classify_error,
    compare_server_config,
    evaluate_prompt_parity,
    extract_vllm_config_metrics,
    generated_prefix_cache_prompt,
    launch_config_from_argv,
    load_gpu_summary,
    load_vllm_metrics_summary,
    parallelism_evidence,
    parse_prometheus_metrics,
    wait_for_server_drain,
)


class ServerEvidenceTests(unittest.TestCase):
    def test_launch_and_metrics_config_match_expected_values(self) -> None:
        launch = launch_config_from_argv(
            [
                "vllm",
                "serve",
                "model",
                "--enable-chunked-prefill",
                "--max-num-batched-tokens",
                "8192",
            ]
        )
        metrics = extract_vllm_config_metrics(
            'vllm:cache_config_info{enable_prefix_caching="True",'
            'gpu_memory_utilization="0.9"} 1\n'
        )
        comparisons = compare_server_config(
            {
                "enable_prefix_caching": "true",
                "enable_chunked_prefill": "true",
                "gpu_memory_utilization": "0.90",
            },
            launch,
            metrics,
        )
        self.assertTrue(all(item["matched"] for item in comparisons))
        self.assertEqual(
            parallelism_evidence(
                {
                    "tensor_parallel_size": "2",
                    "pipeline_parallel_size": "1",
                    "speculative_config": '{"method":"ngram"}',
                }
            )["tensor_parallel_size"],
            "2",
        )

    def test_distributed_errors_are_classified(self) -> None:
        self.assertEqual(
            classify_error("NCCL communicator aborted"),
            "nccl_error",
        )
        self.assertEqual(
            classify_error("worker died during init_process_group"),
            "distributed_worker_error",
        )

    def test_drain_requires_stable_zero_samples(self) -> None:
        samples = iter(
            [
                "vllm:num_requests_running 1\nvllm:num_requests_waiting 1\n",
                "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n",
                "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n",
            ]
        )
        with (
            patch(
                "scripts.benchmark_vllm.fetch_metrics_text",
                side_effect=lambda *_args, **_kwargs: next(samples),
            ),
            patch("scripts.benchmark_vllm.time.sleep"),
        ):
            result = wait_for_server_drain("http://test/metrics", 2, 0.01, 2)
        self.assertTrue(result["drained"])
        self.assertEqual(result["polls"], 3)


class MetricWindowTests(unittest.TestCase):
    def test_counters_use_named_before_and_after_boundaries(self) -> None:
        records = [
            {
                "workload": "w",
                "metrics": {
                    "vllm:prefix_cache_queries_total": 1,
                    "vllm:prefix_cache_hits_total": 1,
                },
            },
            {
                "workload": "w",
                "phase": "before_workload",
                "metrics": {
                    "vllm:prefix_cache_queries_total": 100,
                    "vllm:prefix_cache_hits_total": 40,
                    "vllm:request_prefill_time_seconds_count": 20,
                    "vllm:spec_decode_num_draft_tokens_total": 10,
                    "vllm:spec_decode_num_accepted_tokens_total": 5,
                    "llm_router_retries_total": 0,
                    'llm_router_worker_attempts_total{worker="a"}': 0,
                    'llm_router_worker_attempts_total{worker="b"}': 0,
                    'llm_router_worker_failures_total{worker="a"}': 0,
                    'llm_router_worker_failures_total{worker="b"}': 0,
                },
            },
            {
                "workload": "w",
                "phase": "after_workload",
                "metrics": {
                    "vllm:prefix_cache_queries_total": 160,
                    "vllm:prefix_cache_hits_total": 70,
                    "vllm:request_prefill_time_seconds_count": 28,
                    "vllm:spec_decode_num_draft_tokens_total": 30,
                    "vllm:spec_decode_num_accepted_tokens_total": 17,
                    "llm_router_retries_total": 1,
                    'llm_router_worker_attempts_total{worker="a"}': 8,
                    'llm_router_worker_attempts_total{worker="b"}': 4,
                    'llm_router_worker_failures_total{worker="a"}': 1,
                    'llm_router_worker_failures_total{worker="b"}': 0,
                    'llm_router_worker_circuit_open{worker="a"}': 1,
                    'llm_router_worker_circuit_open{worker="b"}': 0,
                },
            },
            {
                "workload": "w",
                "metrics": {
                    "vllm:prefix_cache_queries_total": 999,
                    "vllm:prefix_cache_hits_total": 999,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )
            summary = load_vllm_metrics_summary(path, "w")
        self.assertEqual(summary["vllm_prefix_cache_query_tokens"], 60)
        self.assertEqual(summary["vllm_prefix_cache_hit_tokens"], 30)
        self.assertEqual(summary["vllm_prefix_cache_hit_rate"], 0.5)
        self.assertEqual(summary["vllm_request_prefill_count"], 8)
        self.assertEqual(summary["vllm_spec_decode_draft_tokens"], 20)
        self.assertEqual(summary["vllm_spec_decode_accepted_tokens"], 12)
        self.assertEqual(summary["vllm_spec_decode_acceptance_rate"], 0.6)
        self.assertEqual(summary["router_retries"], 1)
        self.assertEqual(summary["router_failures"], 1)
        self.assertEqual(summary["router_worker_attempt_imbalance_pct"], 50)
        self.assertEqual(summary["router_circuit_open_workers_max"], 1)

    def test_router_worker_labels_are_preserved(self) -> None:
        metrics = parse_prometheus_metrics(
            'llm_router_worker_attempts_total{worker="replica_a"} 4\n'
            'llm_router_worker_attempts_total{worker="replica_b"} 3\n'
        )
        self.assertEqual(
            metrics[
                'llm_router_worker_attempts_total{worker="replica_a"}'
            ],
            4,
        )

    def test_gpu_imbalance_is_computed_per_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu.csv"
            path.write_text(
                "collected_at,workload,gpu_index,gpu_name,memory_used_mb,"
                "memory_total_mb,gpu_utilization_pct\n"
                "t,w,0,L40S,10000,46000,80\n"
                "t,w,1,L40S,12000,46000,60\n",
                encoding="utf-8",
            )
            summary = load_gpu_summary(path, "w")
        self.assertEqual(summary["gpu_count_observed"], 2)
        self.assertEqual(summary["gpu_memory_used_imbalance_mb"], 2000)
        self.assertEqual(summary["gpu_utilization_imbalance_pct"], 20)


class PrefixPromptTests(unittest.TestCase):
    def test_unique_prefixes_preserve_parity_and_randomize_the_beginning(self) -> None:
        shared = generated_prefix_cache_prompt(
            2048, 1536, 0, "shared", "Summarize."
        )
        unique = [
            generated_prefix_cache_prompt(
                2048, 1536, index, "unique", "Summarize."
            )
            for index in range(100)
        ]
        self.assertTrue(
            all(Counter(prompt.split()) == Counter(shared.split()) for prompt in unique)
        )
        self.assertTrue(all(len(prompt) == len(shared) for prompt in unique))
        self.assertEqual(
            len({tuple(prompt.split()[:2]) for prompt in unique}),
            100,
        )

    def test_observed_prompt_parity_is_enforced(self) -> None:
        rows = [
            {
                "workload": "unique",
                "prompt_parity_group": "g",
                "prompt_parity_max_delta_pct": 2,
                "prompt_tokens_avg": 2048,
            },
            {
                "workload": "shared",
                "prompt_parity_group": "g",
                "prompt_parity_max_delta_pct": 2,
                "prompt_tokens_avg": 2049,
            },
        ]
        checks = evaluate_prompt_parity(rows)
        self.assertTrue(checks[0]["passed"])
        self.assertTrue(rows[0]["prompt_parity_passed"])


if __name__ == "__main__":
    unittest.main()
