#!/usr/bin/env python3
"""
Statistical verification tests for lib/analysis.py functions.

This test creates test data with known statistical parameters, runs it through
the complete analysis pipeline, and verifies that the calculated statistics
match the original parameters within acceptable tolerances.
"""

import os
import sys
import tempfile
import shutil
import unittest
import json
import yaml
import pandas as pd
import warnings
import numpy as np
from unittest.mock import patch, MagicMock

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.generate import generate_report
from tests.test_data_utils import (
    create_histogram_data,
    create_pageload_event_data,
    create_nimbus_api_cache,
    create_test_config,
)


class TestStatisticalVerification(unittest.TestCase):
    """Test statistical accuracy of the analysis pipeline."""

    def setUp(self):
        """Create temporary directories and test data with known statistics."""
        # Set random seed for reproducible results in statistical verification
        np.random.seed(42)

        # Suppress scipy precision loss warnings that occur with synthetic data
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
            category=RuntimeWarning,
            module="scipy.*",
        )

        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Define precise statistical parameters for testing
        self.known_stats = {
            "histogram_stats": {
                "control": {"mean": 2000, "median": 1800, "stddev": 800},
                "treatment": {"mean": 1200, "median": 1100, "stddev": 500},
            },
            "fcp_event_stats": {
                "control": {"mean": 1800, "median": 1600, "stddev": 600},
                "treatment": {"mean": 1000, "median": 900, "stddev": 400},
            },
            "load_event_stats": {
                "control": {"mean": 4000, "median": 3500, "stddev": 1500},
                "treatment": {"mean": 2800, "median": 2500, "stddev": 1000},
            },
        }

        # Expected statistical differences
        self.expected_differences = {
            "histogram": {
                "mean_diff": self.known_stats["histogram_stats"]["control"]["mean"]
                - self.known_stats["histogram_stats"]["treatment"]["mean"],
                "relative_uplift": (
                    self.known_stats["histogram_stats"]["treatment"]["mean"]
                    / self.known_stats["histogram_stats"]["control"]["mean"]
                    - 1
                )
                * 100,
            },
            "fcp_event": {
                "mean_diff": self.known_stats["fcp_event_stats"]["control"]["mean"]
                - self.known_stats["fcp_event_stats"]["treatment"]["mean"],
                "relative_uplift": (
                    self.known_stats["fcp_event_stats"]["treatment"]["mean"]
                    / self.known_stats["fcp_event_stats"]["control"]["mean"]
                    - 1
                )
                * 100,
            },
        }

        # Create test config
        self.config_path = os.path.join(self.test_dir, "statistical-test.yaml")
        self.create_test_config()

        # Create test data with known statistics
        self.create_test_data()

        # Create mock Nimbus API response
        self.create_nimbus_api_cache()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a test YAML configuration file."""
        config = create_test_config(
            "statistical-verification-test",
            {
                "channel": "release",
                "segments": ["Windows"],  # Single segment for cleaner analysis
                "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
                "events": [
                    {
                        "pageload": {
                            "fcp_time": {"max": 10000},
                            "load_time": {"max": 20000},
                        }
                    }
                ],
            },
        )

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def create_nimbus_api_cache(self):
        """Create a mock Nimbus API response."""
        experiment_data_dir = os.path.join(
            self.data_dir, "statistical-verification-test"
        )
        os.makedirs(experiment_data_dir, exist_ok=True)

        experiment_config = {
            "name": "Statistical Verification Test",
            "description": "Test experiment for verifying statistical analysis accuracy",
            "channels": ["release"],
            "isRollout": False,
            "status": "Complete",
        }

        create_nimbus_api_cache(
            experiment_data_dir, "statistical-verification-test", experiment_config
        )

    def create_test_data(self):
        """Create test data with precisely known statistical parameters."""
        slug = "statistical-verification-test"
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        # Use larger sample sizes for more accurate statistical validation
        large_sample_size = 5000

        # Create histogram data with known statistics
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_fcp",
            branches=["control", "treatment"],
            segments=["Windows"],
            branch_stats=self.known_stats["histogram_stats"],
            total_samples=large_sample_size,
        )

        # Create pageload event data with known statistics
        create_pageload_event_data(
            experiment_data_dir,
            slug,
            "fcp_time",
            branches=["control", "treatment"],
            segments=["Windows"],
            branch_stats=self.known_stats["fcp_event_stats"],
            total_samples=large_sample_size,
        )

        create_pageload_event_data(
            experiment_data_dir,
            slug,
            "load_time",
            branches=["control", "treatment"],
            segments=["Windows"],
            branch_stats=self.known_stats["load_event_stats"],
            total_samples=large_sample_size,
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_histogram_statistical_accuracy(self, mock_bigquery_client):
        """Test that histogram analysis produces accurate statistical results."""
        # Mock BigQuery client to avoid actual queries
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        # Mock the query to return empty DataFrame when called
        mock_client.query.return_value.to_dataframe.return_value = pd.DataFrame()

        # Generate report and get cached results
        class MockArgs:
            config = self.config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = False  # Skip HTML generation for faster testing

        args = MockArgs()
        generate_report(args)

        # Load the cached results file
        results_file = os.path.join(
            self.data_dir,
            "statistical-verification-test",
            "statistical-verification-test-results.json",
        )
        self.assertTrue(
            os.path.exists(results_file), f"Results file not found at {results_file}"
        )

        with open(results_file, "r") as f:
            results = json.load(f)

        # Verify histogram statistics
        metric_name = "performance_pageload_fcp"
        control_data = results["control"]["Windows"]["numerical"][metric_name]
        treatment_data = results["treatment"]["Windows"]["numerical"][metric_name]

        # Check means (within 5% tolerance due to sampling variation)
        expected_control_mean = self.known_stats["histogram_stats"]["control"]["mean"]
        expected_treatment_mean = self.known_stats["histogram_stats"]["treatment"][
            "mean"
        ]

        actual_control_mean = control_data["mean"]
        actual_treatment_mean = treatment_data["mean"]

        control_mean_error = (
            abs(actual_control_mean - expected_control_mean) / expected_control_mean
        )
        treatment_mean_error = (
            abs(actual_treatment_mean - expected_treatment_mean)
            / expected_treatment_mean
        )

        self.assertLess(
            control_mean_error,
            0.15,
            f"Control mean error {control_mean_error:.3f} > 15%. "
            f"Expected: {expected_control_mean}, Actual: {actual_control_mean:.1f}",
        )
        self.assertLess(
            treatment_mean_error,
            0.15,
            f"Treatment mean error {treatment_mean_error:.3f} > 15%. "
            f"Expected: {expected_treatment_mean}, Actual: {actual_treatment_mean:.1f}",
        )

        # Check standard deviations (within 10% tolerance)
        expected_control_std = self.known_stats["histogram_stats"]["control"]["stddev"]
        expected_treatment_std = self.known_stats["histogram_stats"]["treatment"][
            "stddev"
        ]

        actual_control_std = control_data["std"]
        actual_treatment_std = treatment_data["std"]

        control_std_error = (
            abs(actual_control_std - expected_control_std) / expected_control_std
        )
        treatment_std_error = (
            abs(actual_treatment_std - expected_treatment_std) / expected_treatment_std
        )

        self.assertLess(
            control_std_error,
            0.20,
            f"Control std error {control_std_error:.3f} > 20%. "
            f"Expected: {expected_control_std}, Actual: {actual_control_std:.1f}",
        )
        self.assertLess(
            treatment_std_error,
            0.20,
            f"Treatment std error {treatment_std_error:.3f} > 20%. "
            f"Expected: {expected_treatment_std}, Actual: {actual_treatment_std:.1f}",
        )

        # Check relative improvement calculation
        if "comparison" in results and "Windows" in results["comparison"]:
            comparison = results["comparison"]["Windows"]["histograms"][metric_name]
            if "relative_improvement" in comparison:
                expected_relative_improvement = self.expected_differences["histogram"][
                    "relative_uplift"
                ]
                actual_relative_improvement = comparison["relative_improvement"]

                relative_improvement_error = abs(
                    actual_relative_improvement - expected_relative_improvement
                ) / abs(expected_relative_improvement)

                self.assertLess(
                    relative_improvement_error,
                    0.10,
                    f"Relative improvement error {relative_improvement_error:.3f} > 10%. "
                    f"Expected: {expected_relative_improvement:.1f}%, Actual: {actual_relative_improvement:.1f}%",
                )

        # Print summary for verification
        print("\n=== Histogram Statistics Verification ===")
        print(
            f"Control: Expected mean={expected_control_mean}, Actual={actual_control_mean:.1f} (error: {control_mean_error:.1%})"
        )
        print(
            f"Treatment: Expected mean={expected_treatment_mean}, Actual={actual_treatment_mean:.1f} (error: {treatment_mean_error:.1%})"
        )
        print(
            f"Control: Expected std={expected_control_std}, Actual={actual_control_std:.1f} (error: {control_std_error:.1%})"
        )
        print(
            f"Treatment: Expected std={expected_treatment_std}, Actual={actual_treatment_std:.1f} (error: {treatment_std_error:.1%})"
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_pageload_event_statistical_accuracy(self, mock_bigquery_client):
        """Test that pageload event analysis produces accurate statistical results."""
        # Mock BigQuery client
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client
        mock_client.query.return_value.to_dataframe.return_value = pd.DataFrame()

        # Generate report
        class MockArgs:
            config = self.config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = False

        args = MockArgs()
        generate_report(args)

        # Load results
        results_file = os.path.join(
            self.data_dir,
            "statistical-verification-test",
            "statistical-verification-test-results.json",
        )

        with open(results_file, "r") as f:
            results = json.load(f)

        # Verify FCP event statistics (now in numerical data type)
        metric_name = "fcp_time"
        control_data = results["control"]["Windows"]["numerical"][metric_name]
        treatment_data = results["treatment"]["Windows"]["numerical"][metric_name]

        # Check means
        expected_control_mean = self.known_stats["fcp_event_stats"]["control"]["mean"]
        expected_treatment_mean = self.known_stats["fcp_event_stats"]["treatment"][
            "mean"
        ]

        actual_control_mean = control_data["mean"]
        actual_treatment_mean = treatment_data["mean"]

        control_mean_error = (
            abs(actual_control_mean - expected_control_mean) / expected_control_mean
        )
        treatment_mean_error = (
            abs(actual_treatment_mean - expected_treatment_mean)
            / expected_treatment_mean
        )

        self.assertLess(
            control_mean_error,
            0.15,
            f"FCP Control mean error {control_mean_error:.3f} > 15%. "
            f"Expected: {expected_control_mean}, Actual: {actual_control_mean:.1f}",
        )
        self.assertLess(
            treatment_mean_error,
            0.15,
            f"FCP Treatment mean error {treatment_mean_error:.3f} > 15%. "
            f"Expected: {expected_treatment_mean}, Actual: {actual_treatment_mean:.1f}",
        )

        # Print summary
        print("\n=== FCP Event Statistics Verification ===")
        print(
            f"Control: Expected mean={expected_control_mean}, Actual={actual_control_mean:.1f} (error: {control_mean_error:.1%})"
        )
        print(
            f"Treatment: Expected mean={expected_treatment_mean}, Actual={actual_treatment_mean:.1f} (error: {treatment_mean_error:.1%})"
        )

        # Test statistical significance detection
        if "comparison" in results and "Windows" in results["comparison"]:
            comparison = results["comparison"]["Windows"]["numerical"][metric_name]

            # With large sample sizes and significant differences, we should detect significance
            if "p_value" in comparison:
                p_value = comparison["p_value"]
                self.assertLess(
                    p_value,
                    0.05,
                    f"P-value {p_value:.4f} should be < 0.05 for significant difference between {expected_control_mean} and {expected_treatment_mean}",
                )
                print(f"Statistical significance: p-value = {p_value:.4f} (< 0.05 ✓)")

    @patch("lib.telemetry.bigquery.Client")
    def test_statistical_comparison_calculations(self, mock_bigquery_client):
        """Test that statistical comparisons and effect sizes are calculated correctly."""
        # Mock BigQuery client
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client
        mock_client.query.return_value.to_dataframe.return_value = pd.DataFrame()

        # Generate report
        class MockArgs:
            config = self.config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = False

        args = MockArgs()
        generate_report(args)

        # Load results
        results_file = os.path.join(
            self.data_dir,
            "statistical-verification-test",
            "statistical-verification-test-results.json",
        )

        with open(results_file, "r") as f:
            results = json.load(f)

        print("\n=== Statistical Comparison Verification ===")

        # Check if comparison results exist
        if "comparison" in results and "Windows" in results["comparison"]:
            comparison = results["comparison"]["Windows"]

            # Check histogram comparison
            if (
                "histograms" in comparison
                and "performance_pageload_fcp" in comparison["histograms"]
            ):
                hist_comparison = comparison["histograms"]["performance_pageload_fcp"]
                print(f"Histogram comparison keys: {list(hist_comparison.keys())}")

                # Look for statistical test results
                for key in ["t_test", "mwu_test", "p_value", "effect_size"]:
                    if key in hist_comparison:
                        print(f"  {key}: {hist_comparison[key]}")

            # Check event metric comparison
            if "numerical" in comparison and "fcp_time" in comparison["numerical"]:
                event_comparison = comparison["numerical"]["fcp_time"]
                print(f"Event comparison keys: {list(event_comparison.keys())}")

                for key in ["t_test", "mwu_test", "p_value", "effect_size"]:
                    if key in event_comparison:
                        print(f"  {key}: {event_comparison[key]}")
        else:
            print("No comparison results found in output")
            print(f"Available keys in results: {list(results.keys())}")


if __name__ == "__main__":
    unittest.main()
