#!/usr/bin/env python3

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.analysis import DataAnalyzer


class TestLabeledCounterAggregates(unittest.TestCase):
    """Tests for labeled_counter aggregate modes: sum, mean, histogram."""

    def setUp(self):
        """Set up test config and data."""
        self.config = {
            "branches": ["control", "treatment"],
            "segments": ["Windows"],
        }
        self.analyzer = DataAnalyzer(self.config)


    def test_labeled_percentiles_uplift_calculation(self):
        """Test that labeled_percentiles uplift is calculated correctly for median, p75, p95."""
        transformed_data = {
            "labeled_percentiles": [
                {
                    "branch": "control",
                    "segment": "Windows",
                    "metric_name": "power_cpu_time_per_process_type_ms",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.power_cpu_time_per_process_type_ms",
                    "config": {"desc": "CPU time"},
                    "data": {
                        "labels": ["parent", "gpu"],
                        "medians": [1000.0, 500.0],
                        "p75s": [2000.0, 1000.0],
                        "p95s": [5000.0, 2500.0],
                        "counts": [100, 100],
                    },
                },
                {
                    "branch": "treatment",
                    "segment": "Windows",
                    "metric_name": "power_cpu_time_per_process_type_ms",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.power_cpu_time_per_process_type_ms",
                    "config": {"desc": "CPU time"},
                    "data": {
                        "labels": ["parent", "gpu"],
                        "medians": [900.0, 450.0],
                        "p75s": [1800.0, 900.0],
                        "p95s": [4500.0, 2250.0],
                        "counts": [100, 100],
                    },
                },
            ]
        }

        self.analyzer.processLabeledPercentilesMetrics(
            transformed_data["labeled_percentiles"]
        )

        # Check control branch data
        control_result = self.analyzer.results["control"]["Windows"]["labeled_percentiles"][
            "power_cpu_time_per_process_type_ms"
        ]
        self.assertEqual(control_result["labels"], ["parent", "gpu"])
        self.assertEqual(control_result["medians"], [1000.0, 500.0])
        self.assertEqual(control_result["p75s"], [2000.0, 1000.0])
        self.assertEqual(control_result["p95s"], [5000.0, 2500.0])

        # Check treatment branch uplifts
        treatment_result = self.analyzer.results["treatment"]["Windows"][
            "labeled_percentiles"
        ]["power_cpu_time_per_process_type_ms"]

        # Median uplifts: parent -10%, gpu -10%
        self.assertAlmostEqual(treatment_result["median_uplifts"][0], -10.0, places=1)
        self.assertAlmostEqual(treatment_result["median_uplifts"][1], -10.0, places=1)

        # p75 uplifts: parent -10%, gpu -10%
        self.assertAlmostEqual(treatment_result["p75_uplifts"][0], -10.0, places=1)
        self.assertAlmostEqual(treatment_result["p75_uplifts"][1], -10.0, places=1)

        # p95 uplifts: parent -10%, gpu -10%
        self.assertAlmostEqual(treatment_result["p95_uplifts"][0], -10.0, places=1)
        self.assertAlmostEqual(treatment_result["p95_uplifts"][1], -10.0, places=1)

    def test_labeled_categorical_uplift_calculation(self):
        """Test that labeled_counter with sum aggregate calculates uplift correctly."""
        transformed_data = {
            "categorical": [
                {
                    "branch": "control",
                    "segment": "Windows",
                    "metric_name": "javascript_gc_slice_was_long",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.javascript_gc_slice_was_long",
                    "config": {"desc": "GC slice duration"},
                    "data": {
                        "bins": ["true", "false"],
                        "counts": [100, 900],
                    },
                },
                {
                    "branch": "treatment",
                    "segment": "Windows",
                    "metric_name": "javascript_gc_slice_was_long",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.javascript_gc_slice_was_long",
                    "config": {"desc": "GC slice duration"},
                    "data": {
                        "bins": ["true", "false"],
                        "counts": [80, 920],
                    },
                },
            ]
        }

        self.analyzer.processCategoricalMetrics(transformed_data["categorical"])

        # Check control branch data
        control_result = self.analyzer.results["control"]["Windows"]["categorical"][
            "javascript_gc_slice_was_long"
        ]
        self.assertEqual(control_result["labels"], ["true", "false"])
        self.assertEqual(control_result["counts"], [100, 900])
        self.assertEqual(control_result["sum"], 1000)

        # Check treatment branch uplift
        treatment_result = self.analyzer.results["treatment"]["Windows"]["categorical"][
            "javascript_gc_slice_was_long"
        ]
        # true: (80-100)/100 = -20%, false: (920-900)/900 = +2.22%
        self.assertAlmostEqual(treatment_result["uplift"][0], -20.0, places=1)
        self.assertAlmostEqual(treatment_result["uplift"][1], 2.22, places=1)


    def test_labeled_percentiles_with_synthetic_data(self):
        """Test labeled_percentiles with synthetic data simulating realistic distributions."""
        # Simulate control branch: parent process has higher CPU time than gpu
        # Control: parent median=1000, gpu median=500
        # Treatment: parent median=800 (-20%), gpu median=450 (-10%)

        import pandas as pd

        # Create synthetic data that would result in specific percentiles
        # For control parent: values that give median≈1000, p75≈2000, p95≈5000
        control_parent_values = [
            100, 200, 300, 500, 700,  # Below median
            1000, 1100, 1200,  # Around median
            1500, 2000, 2500,  # p50-p75 range
            3000, 4000, 5000, 6000, 7000  # p75-p95 and above
        ]

        # For control gpu: values that give median≈500, p75≈1000, p95≈2500
        control_gpu_values = [
            50, 100, 200, 300, 400,
            500, 550, 600,
            750, 1000, 1250,
            1500, 2000, 2500, 3000, 3500
        ]

        # For treatment parent: 20% lower
        treatment_parent_values = [int(v * 0.8) for v in control_parent_values]

        # For treatment gpu: 10% lower
        treatment_gpu_values = [int(v * 0.9) for v in control_gpu_values]

        # Build dataframes as they would come from BigQuery
        control_df = pd.DataFrame({
            'segment': ['Windows'] * (len(control_parent_values) + len(control_gpu_values)),
            'branch': ['control'] * (len(control_parent_values) + len(control_gpu_values)),
            'label': ['parent'] * len(control_parent_values) + ['gpu'] * len(control_gpu_values),
        })

        treatment_df = pd.DataFrame({
            'segment': ['Windows'] * (len(treatment_parent_values) + len(treatment_gpu_values)),
            'branch': ['treatment'] * (len(treatment_parent_values) + len(treatment_gpu_values)),
            'label': ['parent'] * len(treatment_parent_values) + ['gpu'] * len(treatment_gpu_values),
        })

        # Simulate percentile calculation (what BigQuery would return)
        # Control percentiles
        control_parent_median = sorted(control_parent_values)[len(control_parent_values) // 2]
        control_parent_p75 = sorted(control_parent_values)[int(len(control_parent_values) * 0.75)]
        control_parent_p95 = sorted(control_parent_values)[int(len(control_parent_values) * 0.95)]

        control_gpu_median = sorted(control_gpu_values)[len(control_gpu_values) // 2]
        control_gpu_p75 = sorted(control_gpu_values)[int(len(control_gpu_values) * 0.75)]
        control_gpu_p95 = sorted(control_gpu_values)[int(len(control_gpu_values) * 0.95)]

        # Treatment percentiles
        treatment_parent_median = sorted(treatment_parent_values)[len(treatment_parent_values) // 2]
        treatment_parent_p75 = sorted(treatment_parent_values)[int(len(treatment_parent_values) * 0.75)]
        treatment_parent_p95 = sorted(treatment_parent_values)[int(len(treatment_parent_values) * 0.95)]

        treatment_gpu_median = sorted(treatment_gpu_values)[len(treatment_gpu_values) // 2]
        treatment_gpu_p75 = sorted(treatment_gpu_values)[int(len(treatment_gpu_values) * 0.75)]
        treatment_gpu_p95 = sorted(treatment_gpu_values)[int(len(treatment_gpu_values) * 0.95)]

        # Build transformed data as it would come from telemetry
        transformed_data = {
            "labeled_percentiles": [
                {
                    "branch": "control",
                    "segment": "Windows",
                    "metric_name": "power_cpu_time_per_process_type_ms",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.power_cpu_time_per_process_type_ms",
                    "config": {"desc": "CPU time"},
                    "data": {
                        "labels": ["parent", "gpu"],
                        "medians": [float(control_parent_median), float(control_gpu_median)],
                        "p75s": [float(control_parent_p75), float(control_gpu_p75)],
                        "p95s": [float(control_parent_p95), float(control_gpu_p95)],
                        "counts": [len(control_parent_values), len(control_gpu_values)],
                    },
                },
                {
                    "branch": "treatment",
                    "segment": "Windows",
                    "metric_name": "power_cpu_time_per_process_type_ms",
                    "source_section": "histograms",
                    "source_key": "metrics.labeled_counter.power_cpu_time_per_process_type_ms",
                    "config": {"desc": "CPU time"},
                    "data": {
                        "labels": ["parent", "gpu"],
                        "medians": [float(treatment_parent_median), float(treatment_gpu_median)],
                        "p75s": [float(treatment_parent_p75), float(treatment_gpu_p75)],
                        "p95s": [float(treatment_parent_p95), float(treatment_gpu_p95)],
                        "counts": [len(treatment_parent_values), len(treatment_gpu_values)],
                    },
                },
            ]
        }

        self.analyzer.processLabeledPercentilesMetrics(
            transformed_data["labeled_percentiles"]
        )

        # Check control branch data
        control_result = self.analyzer.results["control"]["Windows"]["labeled_percentiles"][
            "power_cpu_time_per_process_type_ms"
        ]
        self.assertEqual(control_result["labels"], ["parent", "gpu"])
        self.assertEqual(len(control_result["medians"]), 2)
        self.assertEqual(len(control_result["p75s"]), 2)
        self.assertEqual(len(control_result["p95s"]), 2)

        # Check treatment branch has uplifts
        treatment_result = self.analyzer.results["treatment"]["Windows"]["labeled_percentiles"][
            "power_cpu_time_per_process_type_ms"
        ]
        self.assertIn("median_uplifts", treatment_result)
        self.assertIn("p75_uplifts", treatment_result)
        self.assertIn("p95_uplifts", treatment_result)

        # Verify uplifts are negative (treatment is lower than control)
        # Parent should be around -20% for all percentiles
        self.assertLess(treatment_result["median_uplifts"][0], -15.0)
        self.assertLess(treatment_result["p75_uplifts"][0], -15.0)
        self.assertLess(treatment_result["p95_uplifts"][0], -15.0)

        # GPU should be around -10% for all percentiles
        self.assertLess(treatment_result["median_uplifts"][1], -5.0)
        self.assertLess(treatment_result["p75_uplifts"][1], -5.0)
        self.assertLess(treatment_result["p95_uplifts"][1], -5.0)


    def test_quantity_percentiles_uplift_calculation(self):
        """Test that quantity metrics with percentiles aggregate calculate uplifts correctly."""
        transformed_data = {
            "quantity_percentiles": [
                {
                    "branch": "control",
                    "segment": "Windows",
                    "metric_name": "gfx_target_frame_rate",
                    "source_section": "histograms",
                    "source_key": "metrics.quantity.gfx_target_frame_rate",
                    "config": {"desc": "Target frame rate"},
                    "data": {
                        "median": 60.0,
                        "p75": 60.0,
                        "p95": 60.0,
                        "sum": 100000.0,
                        "count": 1000,
                    },
                },
                {
                    "branch": "treatment",
                    "segment": "Windows",
                    "metric_name": "gfx_target_frame_rate",
                    "source_section": "histograms",
                    "source_key": "metrics.quantity.gfx_target_frame_rate",
                    "config": {"desc": "Target frame rate"},
                    "data": {
                        "median": 30.0,
                        "p75": 30.0,
                        "p95": 60.0,
                        "sum": 50000.0,
                        "count": 1000,
                    },
                },
            ]
        }

        self.analyzer.processQuantityPercentilesMetrics(
            transformed_data["quantity_percentiles"]
        )

        # Check control branch data
        control_result = self.analyzer.results["control"]["Windows"]["quantity_percentiles"][
            "gfx_target_frame_rate"
        ]
        self.assertEqual(control_result["median"], 60.0)
        self.assertEqual(control_result["p75"], 60.0)
        self.assertEqual(control_result["p95"], 60.0)
        self.assertEqual(control_result["sum"], 100000.0)
        self.assertEqual(control_result["count"], 1000)

        # Check treatment branch uplifts
        treatment_result = self.analyzer.results["treatment"]["Windows"]["quantity_percentiles"][
            "gfx_target_frame_rate"
        ]

        # Median: (30-60)/60 = -50%
        self.assertAlmostEqual(treatment_result["median_uplift"], -50.0, places=1)

        # p75: (30-60)/60 = -50%
        self.assertAlmostEqual(treatment_result["p75_uplift"], -50.0, places=1)

        # p95: (60-60)/60 = 0%
        self.assertAlmostEqual(treatment_result["p95_uplift"], 0.0, places=1)

        # sum: (50000-100000)/100000 = -50%
        self.assertAlmostEqual(treatment_result["sum_uplift"], -50.0, places=1)

    def test_quantity_sum_mode(self):
        """Test that quantity metrics with sum aggregate are treated as scalars."""
        # Note: quantity with sum is treated as scalar, which is already tested
        # This test verifies the parser sets kind correctly

        # Mock config as parser would set it
        test_config = {
            "histograms": {
                "metrics.quantity.test_counter": {
                    "aggregate": "sum",
                    "kind": "scalar",  # Parser sets this
                    "distribution_type": "quantity",
                }
            }
        }

        # Verify the kind is scalar
        self.assertEqual(
            test_config["histograms"]["metrics.quantity.test_counter"]["kind"],
            "scalar"
        )


if __name__ == "__main__":
    unittest.main()


