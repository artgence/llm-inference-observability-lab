#!/usr/bin/env python3
"""Tests for Month 5 scaling calculations and report output."""

from __future__ import annotations

import unittest

from scripts.analyze_month5 import add_scaling_metrics, build_report


class Month5AnalysisTests(unittest.TestCase):
    def test_scaling_efficiency_uses_gpu_count(self) -> None:
        rows = [
            {
                "workload": "same",
                "deployment_label": "single",
                "deployment_type": "single_gpu",
                "deployment_gpu_count": 1,
                "output_tokens_per_sec": 100,
            },
            {
                "workload": "same",
                "deployment_label": "tp2",
                "deployment_type": "tensor_parallel",
                "deployment_gpu_count": 2,
                "output_tokens_per_sec": 150,
            },
        ]
        add_scaling_metrics(rows)
        self.assertEqual(rows[1]["throughput_speedup"], 1.5)
        self.assertEqual(rows[1]["scaling_efficiency"], 0.75)

    def test_report_keeps_sharding_replication_decision(self) -> None:
        rows = [
            {
                "workload": "same",
                "deployment_label": "single",
                "deployment_type": "single_gpu",
                "deployment_gpu_count": 1,
                "tensor_parallel_size": 1,
                "output_tokens_per_sec": 100,
                "server_config_verified": True,
                "server_launch_command": "vllm serve model",
            }
        ]
        report = build_report(rows)
        self.assertIn("Use TP when model-fit", report)
        self.assertIn("Prefer independent replicas", report)


if __name__ == "__main__":
    unittest.main()
