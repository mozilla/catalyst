"""
Comprehensive end-to-end test cases with various experiment configurations.

This test suite creates 5 different experiment scenarios to test:
1. Standard A/B experiment with histograms and pageload events
2. Rollout with single branch and crash events
3. Multi-branch experiment with scalar metrics only
4. Complex experiment with mixed metric types across multiple segments
5. Desktop-only experiment with memory and performance metrics

Each test creates synthetic data, runs the full pipeline, and validates the HTML output.
"""

import os
import sys
import tempfile
import shutil
import unittest
import warnings
from unittest.mock import patch, MagicMock
import yaml
from bs4 import BeautifulSoup

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.generate import generate_report
from tests.test_data_utils import (
    create_histogram_data,
    create_pageload_event_data,
    create_crash_event_data,
    create_nimbus_api_cache,
)


class TestComprehensiveEndToEnd(unittest.TestCase):
    """Comprehensive end-to-end tests with various experiment configurations."""

    def setUp(self):
        """Create temporary directories for tests."""
        # Suppress scipy precision loss warnings that occur with synthetic data
        # These are not errors but informationals about statistical precision
        warnings.filterwarnings(
            "ignore",
            message="Precision loss occurred in moment calculation due to catastrophic cancellation.*",
            category=RuntimeWarning,
            module="scipy.*"
        )

        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def _create_config_file(self, config_data, filename):
        """Helper to create a YAML config file."""
        config_path = os.path.join(self.test_dir, filename)
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        return config_path

    def _validate_html_contains_metrics(
        self, html_path, expected_metrics, expected_segments
    ):
        """Helper to validate that HTML contains expected metrics and segments."""
        with open(html_path, "r") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        # Check that all expected segments are present
        for segment in expected_segments:
            self.assertIn(segment, html_content, f"Segment {segment} should be in HTML")

        # Check that all expected metrics are present
        for metric in expected_metrics:
            # Look for metric sections or links
            metric_found = (
                soup.find("div", {"id": lambda x: x and metric in str(x)}) is not None
                or soup.find("a", href=lambda x: x and metric in str(x)) is not None
                or metric in html_content
            )
            self.assertTrue(metric_found, f"Metric {metric} should be in HTML")

    @patch("lib.telemetry.bigquery.Client")
    def test_standard_ab_experiment_with_histograms_and_pageload(
        self, mock_bigquery_client
    ):
        """Test Case 1: Standard A/B experiment with histograms and pageload events."""
        # Mock BigQuery client
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "standard-ab-test"

        # Create experiment config
        config_data = {
            "slug": slug,
            "segments": ["Windows", "Mac", "Linux"],
            "histograms": [
                "metrics.timing_distribution.performance_pageload_fcp",
                "metrics.memory_distribution.memory_total",
            ],
            "events": [
                {
                    "pageload": {
                        "fcp_time": {"max": 10000},
                        "lcp_time": {"max": 15000},
                        "load_time": {"max": 20000},
                    }
                }
            ],
            "max_parallel_queries": 6,
        }

        config_path = self._create_config_file(config_data, f"{slug}.yaml")

        # Create experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        # Create Nimbus API cache
        experiment_config = {
            "name": "Standard A/B Test",
            "description": "Testing FCP and memory improvements",
            "branches": [
                {"name": "control", "slug": "control", "description": "Control branch"},
                {
                    "name": "treatment",
                    "slug": "treatment",
                    "description": "Treatment branch",
                },
            ],
            "startDate": "2024-01-01",
            "endDate": "2024-01-31",
            "channels": ["beta"],
            "isRollout": False,
            "status": "Complete",
        }
        create_nimbus_api_cache(experiment_data_dir, slug, experiment_config)

        # Create synthetic histogram data with meaningful differences
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_fcp",
            branches=["control", "treatment"],
            segments=["Windows", "Mac", "Linux"],
            branch_stats={
                "control": {"mean": 2500, "median": 2000, "stddev": 1000},
                "treatment": {"mean": 1800, "median": 1500, "stddev": 800},
            },
            total_samples=5000,
        )

        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.memory_distribution.memory_total",
            branches=["control", "treatment"],
            segments=["Windows", "Mac", "Linux"],
            branch_stats={
                "control": {"mean": 8000000, "median": 7500000, "stddev": 2000000},
                "treatment": {"mean": 7200000, "median": 6800000, "stddev": 1800000},
            },
            total_samples=3000,
        )

        # Create pageload event data
        for metric in ["fcp_time", "lcp_time", "load_time"]:
            create_pageload_event_data(
                experiment_data_dir,
                slug,
                metric,
                branches=["control", "treatment"],
                segments=["Windows", "Mac", "Linux"],
                total_samples=8000,
            )

        # Run report generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        generate_report(MockArgs())

        # Validate output
        html_path = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")

        self._validate_html_contains_metrics(
            html_path,
            expected_metrics=[
                "performance_pageload_fcp",
                "memory_total",
                "fcp_time",
                "lcp_time",
                "load_time",
            ],
            expected_segments=["Windows", "Mac", "Linux"],
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_rollout_with_crash_events(self, mock_bigquery_client):
        """Test Case 2: Rollout experiment with crash events."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "crash-reduction-rollout"

        # Create rollout config with single branch
        config_data = {
            "slug": slug,
            "segments": ["Windows", "Mac"],
            "events": ["crash"],
            "max_parallel_queries": 2,
        }

        config_path = self._create_config_file(config_data, f"{slug}.yaml")

        # Create Nimbus API cache for rollout
        experiment_config = {
            "name": "Crash Reduction Rollout",
            "description": "Rolling out crash fixes",
            "branches": [
                {"name": "default", "slug": "default", "description": "Default branch"},
                {
                    "name": "rollout",
                    "slug": "rollout",
                    "description": "Rollout branch with fixes",
                },
            ],
            "startDate": "2024-02-01",
            "endDate": "2024-02-08",
            "channels": ["beta"],
            "isRollout": True,
            "status": "Complete",
        }

        # Create experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        create_nimbus_api_cache(experiment_data_dir, slug, experiment_config)

        # Create crash data showing improvement
        create_crash_event_data(
            experiment_data_dir,
            slug,
            metric_name="total_crashes",
            branches=["default", "rollout"],
            segments=["Windows", "Mac"],
            crash_counts={
                "default": {"Windows": 2500, "Mac": 800},
                "rollout": {"Windows": 1800, "Mac": 600},  # 28% and 25% reduction
            },
        )

        # Run report generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        generate_report(MockArgs())

        # Validate output
        html_path = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")

        self._validate_html_contains_metrics(
            html_path,
            expected_metrics=["total_crashes"],
            expected_segments=["Windows", "Mac"],
        )

        # Validate crash reduction is visible
        with open(html_path, "r") as f:
            html_content = f.read()
        self.assertIn("total_crashes", html_content)

    @patch("lib.telemetry.bigquery.Client")
    def test_multi_branch_scalar_only(self, mock_bigquery_client):
        """Test Case 3: Multi-branch experiment with scalar metrics only."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "multi-branch-crashes"

        # Create config with 3 branches and only scalar metrics
        config_data = {
            "slug": slug,
            "segments": ["Android"],
            "events": ["crash"],
            "max_parallel_queries": 3,
        }

        config_path = self._create_config_file(config_data, f"{slug}.yaml")

        # Create Nimbus API cache for multi-branch
        experiment_config = {
            "name": "Multi-Branch Crash Test",
            "description": "Testing multiple crash mitigation strategies",
            "branches": [
                {"name": "control", "slug": "control", "description": "Control branch"},
                {
                    "name": "strategy-a",
                    "slug": "strategy-a",
                    "description": "Strategy A",
                },
                {
                    "name": "strategy-b",
                    "slug": "strategy-b",
                    "description": "Strategy B",
                },
            ],
            "startDate": "2024-03-01",
            "endDate": "2024-03-31",
            "channels": ["nightly"],
            "isRollout": False,
            "status": "Complete",
        }

        # Create experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        create_nimbus_api_cache(experiment_data_dir, slug, experiment_config)

        # Create crash data for 3 branches
        create_crash_event_data(
            experiment_data_dir,
            slug,
            metric_name="total_crashes",
            branches=["control", "strategy-a", "strategy-b"],
            segments=["Android"],
            crash_counts={
                "control": {"Android": 5000},
                "strategy-a": {"Android": 4200},  # 16% reduction
                "strategy-b": {"Android": 3800},  # 24% reduction
            },
        )

        # Run report generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        generate_report(MockArgs())

        # Validate output
        html_path = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")

        self._validate_html_contains_metrics(
            html_path, expected_metrics=["total_crashes"], expected_segments=["Android"]
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_complex_mixed_metrics_multi_segment(self, mock_bigquery_client):
        """Test Case 4: Complex experiment with mixed metric types across multiple segments."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "complex-mixed-metrics"

        # Create config with all metric types
        config_data = {
            "slug": slug,
            "segments": ["Windows", "Linux", "Mac", "Android"],
            "histograms": [
                "metrics.timing_distribution.performance_pageload_fcp",
                "metrics.timing_distribution.perf_largest_contentful_paint",
                "metrics.memory_distribution.memory_total",
            ],
            "events": [
                "crash",
                {"pageload": {"fcp_time": {"max": 8000}, "lcp_time": {"max": 12000}}},
            ],
            "max_parallel_queries": 8,
        }

        config_path = self._create_config_file(config_data, f"{slug}.yaml")

        # Create Nimbus API cache
        experiment_config = {
            "name": "Complex Mixed Metrics Test",
            "description": "Testing comprehensive performance improvements",
            "branches": [
                {"name": "control", "slug": "control", "description": "Control branch"},
                {
                    "name": "treatment",
                    "slug": "treatment",
                    "description": "Performance optimizations",
                },
            ],
            "startDate": "2024-04-01",
            "endDate": "2024-04-08",
            "channels": ["beta"],
            "isRollout": False,
            "status": "Complete",
        }

        # Create experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        create_nimbus_api_cache(experiment_data_dir, slug, experiment_config)

        # Create histogram data for multiple metrics
        histogram_metrics = [
            ("metrics.timing_distribution.performance_pageload_fcp", "fcp", 2000, 1600),
            (
                "metrics.timing_distribution.perf_largest_contentful_paint",
                "lcp",
                3000,
                2400,
            ),
            ("metrics.memory_distribution.memory_total", "memory", 8000000, 7500000),
        ]

        for full_name, short_name, control_mean, treatment_mean in histogram_metrics:
            create_histogram_data(
                experiment_data_dir,
                slug,
                full_name,
                branches=["control", "treatment"],
                segments=["Windows", "Linux", "Mac", "Android"],
                branch_stats={
                    "control": {
                        "mean": control_mean,
                        "median": control_mean * 0.8,
                        "stddev": control_mean * 0.3,
                    },
                    "treatment": {
                        "mean": treatment_mean,
                        "median": treatment_mean * 0.8,
                        "stddev": treatment_mean * 0.3,
                    },
                },
                total_samples=4000,
            )

        # Create pageload event data
        for metric in ["fcp_time", "lcp_time"]:
            create_pageload_event_data(
                experiment_data_dir,
                slug,
                metric,
                branches=["control", "treatment"],
                segments=["Windows", "Linux", "Mac", "Android"],
                total_samples=6000,
            )

        # Create crash data
        create_crash_event_data(
            experiment_data_dir,
            slug,
            metric_name="total_crashes",
            branches=["control", "treatment"],
            segments=["Windows", "Linux", "Mac", "Android"],
            crash_counts={
                "control": {
                    "Windows": 3000,
                    "Linux": 1500,
                    "Mac": 800,
                    "Android": 2000,
                },
                "treatment": {
                    "Windows": 2600,
                    "Linux": 1200,
                    "Mac": 650,
                    "Android": 1700,
                },
            },
        )

        # Run report generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        generate_report(MockArgs())

        # Validate output
        html_path = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")

        self._validate_html_contains_metrics(
            html_path,
            expected_metrics=[
                "performance_pageload_fcp",
                "perf_largest_contentful_paint",
                "memory_total",
                "fcp_time",
                "lcp_time",
                "total_crashes",
            ],
            expected_segments=["Windows", "Linux", "Mac", "Android"],
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_desktop_performance_optimization(self, mock_bigquery_client):
        """Test Case 5: Desktop-only experiment with memory and performance metrics."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "desktop-perf-optimization"

        # Create desktop-focused config
        config_data = {
            "slug": slug,
            "segments": ["Windows", "Mac", "Linux"],
            "histograms": [
                "metrics.memory_distribution.memory_total",
                "metrics.timing_distribution.performance_pageload_fcp",
                "metrics.timing_distribution.performance_pageload_load_time",
            ],
            "events": [
                {
                    "pageload": {
                        "fcp_time": {"max": 5000},
                        "load_time": {"max": 10000},
                        "response_time": {"max": 3000},
                    }
                }
            ],
            "max_parallel_queries": 6,
        }

        config_path = self._create_config_file(config_data, f"{slug}.yaml")

        # Create Nimbus API cache
        experiment_config = {
            "name": "Desktop Performance Optimization",
            "description": "Optimizing memory usage and page load times on desktop",
            "branches": [
                {"name": "control", "slug": "control", "description": "Control branch"},
                {
                    "name": "optimized",
                    "slug": "optimized",
                    "description": "Memory and performance optimizations",
                },
            ],
            "startDate": "2024-05-01",
            "endDate": "2024-05-31",
            "channels": ["nightly", "beta"],
            "isRollout": False,
            "status": "Complete",
        }

        # Create experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        create_nimbus_api_cache(experiment_data_dir, slug, experiment_config)

        # Create histogram data with significant improvements
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.memory_distribution.memory_total",
            branches=["control", "optimized"],
            segments=["Windows", "Mac", "Linux"],
            branch_stats={
                "control": {"mean": 12000000, "median": 10000000, "stddev": 4000000},
                "optimized": {
                    "mean": 9500000,
                    "median": 8000000,
                    "stddev": 3000000,
                },  # 21% reduction
            },
            total_samples=10000,
        )

        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_fcp",
            branches=["control", "optimized"],
            segments=["Windows", "Mac", "Linux"],
            branch_stats={
                "control": {"mean": 1800, "median": 1400, "stddev": 800},
                "optimized": {
                    "mean": 1200,
                    "median": 1000,
                    "stddev": 600,
                },  # 33% improvement
            },
            total_samples=15000,
        )

        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_load_time",
            branches=["control", "optimized"],
            segments=["Windows", "Mac", "Linux"],
            branch_stats={
                "control": {"mean": 4500, "median": 3800, "stddev": 2000},
                "optimized": {
                    "mean": 3200,
                    "median": 2800,
                    "stddev": 1500,
                },  # 29% improvement
            },
            total_samples=12000,
        )

        # Create pageload event data
        pageload_metrics = [
            ("fcp_time", 5000),
            ("load_time", 10000),
            ("response_time", 3000),
        ]

        for metric, max_val in pageload_metrics:
            create_pageload_event_data(
                experiment_data_dir,
                slug,
                metric,
                branches=["control", "optimized"],
                segments=["Windows", "Mac", "Linux"],
                total_samples=20000,
            )

        # Run report generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        generate_report(MockArgs())

        # Validate output
        html_path = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(html_path), "HTML report should be generated")

        self._validate_html_contains_metrics(
            html_path,
            expected_metrics=[
                "memory_total",
                "performance_pageload_fcp",
                "performance_pageload_load_time",
                "fcp_time",
                "load_time",
                "response_time",
            ],
            expected_segments=["Windows", "Mac", "Linux"],
        )

        # Validate performance improvements are visible
        with open(html_path, "r") as f:
            html_content = f.read()

        # Should contain the optimized branch
        self.assertIn("optimized", html_content)


if __name__ == "__main__":
    unittest.main()
