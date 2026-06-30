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
    compare_server_config,
    evaluate_prompt_parity,
    extract_vllm_config_metrics,
    generated_prefix_cache_prompt,
    launch_config_from_argv,
    load_vllm_metrics_summary,
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
                },
            },
            {
                "workload": "w",
                "phase": "after_workload",
                "metrics": {
                    "vllm:prefix_cache_queries_total": 160,
                    "vllm:prefix_cache_hits_total": 70,
                    "vllm:request_prefill_time_seconds_count": 28,
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
