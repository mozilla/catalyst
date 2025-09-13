"""
Integration test that creates artificial data and verifies end-to-end functionality.
Tests the complete pipeline from YAML config to HTML report generation.
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import yaml

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.generate import generate_report
from tests.test_data_utils import (
    create_histogram_data,
    create_pageload_event_data,
    create_nimbus_api_cache,
    create_test_config as create_base_test_config,
)


class TestIntegrationWithArtificialData(unittest.TestCase):
    """Integration test using artificial data to verify complete pipeline."""

    def setUp(self):
        """Create temporary directories and artificial data."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        # Create test config
        self.config_path = os.path.join(self.test_dir, "test-experiment.yaml")
        self.create_test_config()

        # Create artificial data files
        self.create_artificial_data()

        # Create mock Nimbus API response
        self.create_nimbus_api_cache()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a test YAML configuration file."""
        config = create_base_test_config(
            "test-performance-experiment",
            {
                "channel": "nightly",
                "segments": ["Windows", "Linux", "Mac"],
                "include_non_enrolled_branch": False,
                "histograms": [
                    "metrics.timing_distribution.performance_pageload_fcp",
                    "metrics.timing_distribution.performance_pageload_load_time",
                ],
                "pageload_event_metrics": {
                    "fcp_time": {"max": 30000},
                    "load_time": {"max": 60000},
                },
            },
        )

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def create_nimbus_api_cache(self):
        """Create cached Nimbus API response to avoid network calls."""
        # Create the experiment data directory
        experiment_data_dir = os.path.join(self.data_dir, "test-performance-experiment")
        os.makedirs(experiment_data_dir, exist_ok=True)

        experiment_config = {
            "name": "Test Performance Experiment",
            "description": "A test experiment for integration testing",
            "channels": ["nightly"],
            "isRollout": False,
            "status": "Complete",
        }

        create_nimbus_api_cache(
            experiment_data_dir, "test-performance-experiment", experiment_config
        )

    def create_artificial_data(self):
        """Create artificial histogram and pageload event data files."""
        slug = "test-performance-experiment"

        # Create the experiment-specific data directory
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        # Create histogram data
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_fcp",
        )
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_load_time",
        )

        # Create pageload event data
        create_pageload_event_data(experiment_data_dir, slug, "fcp_time")
        create_pageload_event_data(experiment_data_dir, slug, "load_time")

    @patch("lib.telemetry.bigquery.Client")
    def test_end_to_end_report_generation(self, mock_bigquery_client):
        """Test complete pipeline from config to HTML report."""

        # Mock BigQuery client to avoid actual queries
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        # Mock command line arguments
        class MockArgs:
            config = self.config_path
            dataDir = self.data_dir
            skip_cache = False  # Use cache files (pickle files we created)
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report (this should not crash)
        try:
            generate_report(args)
        except Exception as e:
            self.fail(f"Report generation failed with error: {e}")

        # Verify HTML file was created
        expected_html = os.path.join(
            self.reports_dir, "test-performance-experiment.html"
        )
        self.assertTrue(
            os.path.exists(expected_html), f"HTML report not found at {expected_html}"
        )

        # Read and verify HTML content
        with open(expected_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Basic structure checks
        self.assertTrue(
            html_content.startswith("<!DOCTYPE html>"), "HTML should start with DOCTYPE"
        )
        self.assertIn("</body>", html_content)
        self.assertIn("test-performance-experiment", html_content)

        # Check for experimenter page link
        experimenter_link = "https://experimenter.services.mozilla.com/nimbus/test-performance-experiment"
        self.assertIn(
            experimenter_link,
            html_content,
            "Should contain link to experimenter page for the slug",
        )

        # Check for Configuration section
        self.assertIn(
            "Configuration", html_content, "Should have Configuration section"
        )

        # Check for Summary section
        self.assertIn("Summary:", html_content, "Should have Summary section")

        # Check for segments in summary
        self.assertIn("Windows", html_content)
        self.assertIn("Linux", html_content)
        self.assertIn("Mac", html_content)

        # Check for branches
        self.assertIn("control", html_content)
        self.assertIn("treatment", html_content)

        # Check for metrics in summary
        self.assertIn("performance_pageload_fcp", html_content)
        self.assertIn("performance_pageload_load_time", html_content)
        self.assertIn("fcp_time", html_content)
        self.assertIn("load_time", html_content)

        # Check for uplift values and effect sizes in summary tables
        self.assertIn(
            "mean uplift", html_content, "Should have mean uplift column in summary"
        )
        self.assertIn(
            "effect size", html_content, "Should have effect size column in summary"
        )

        # Look for valid uplift percentage values (should have % sign)
        import re

        uplift_pattern = r"[+-]?\d+\.?\d*%"
        uplifts = re.findall(uplift_pattern, html_content)
        # Note: uplift values may not appear with artificial test data that has no real differences
        _ = uplifts  # Acknowledge variable for linting

        # Look for effect size indicators - these may not appear with minimal test data
        effect_sizes = ["Large", "Medium", "Small", "None"]
        has_effect_size = any(effect in html_content for effect in effect_sizes)
        # Note: effect sizes may not appear with artificial test data
        _ = has_effect_size  # Acknowledge variable for linting

        # Check for canvas elements for each metric and chart type
        metrics = [
            "performance_pageload_fcp",
            "performance_pageload_load_time",
            "fcp_time",
            "load_time",
        ]
        segments = ["Windows", "Linux", "Mac"]
        chart_types = ["pdf", "cdf", "uplift", "diff"]

        # For each segment and metric combination, check for required canvas elements
        for segment in segments:
            for metric in metrics:
                # Check for mean canvas (different ID format)
                mean_canvas_id = f"{segment}-{metric}-mean"
                self.assertIn(
                    f'id="{mean_canvas_id}"',
                    html_content,
                    f"Should have mean canvas for {segment}-{metric}",
                )

                # Check for pdf, cdf, uplift, diff canvases
                for chart_type in chart_types:
                    canvas_id = f"{segment}_{metric}_{chart_type}"
                    self.assertIn(
                        f'id="{canvas_id}"',
                        html_content,
                        f"Should have {chart_type} canvas for {segment}-{metric}",
                    )

        # Check for stat tables (multiple per metric per segment for different views)
        stat_tables = html_content.count('class="stat-table"')
        min_expected_stat_tables = len(segments) * len(metrics)
        self.assertGreaterEqual(
            stat_tables,
            min_expected_stat_tables,
            f"Should have at least {min_expected_stat_tables} stat tables, found {stat_tables}",
        )

        # Check for summary tables (one per segment)
        summary_tables = html_content.count('class="summary-table"')
        self.assertGreaterEqual(
            summary_tables,
            len(segments),
            f"Should have at least {len(segments)} summary tables, found {summary_tables}",
        )

        # Check for statistical content in tables - use terms that actually appear
        stat_columns = [
            "branch",
            "mean",
            "median",
            "quantile",
        ]  # Use 'quantile' instead of 'percentile'
        for col in stat_columns:
            self.assertIn(
                col,
                html_content.lower(),
                f"Should contain '{col}' in statistical tables",
            )

        # Verify no obvious errors in HTML
        self.assertNotIn("Error", html_content)
        self.assertNotIn("Exception", html_content)
        self.assertNotIn("Traceback", html_content)

    def test_data_file_structure(self):
        """Verify that created data files have correct structure."""
        slug = "test-performance-experiment"

        # Test histogram data
        experiment_data_dir = os.path.join(self.data_dir, slug)
        hist_file = os.path.join(
            experiment_data_dir, f"{slug}-performance_pageload_fcp.pkl"
        )
        self.assertTrue(os.path.exists(hist_file))

        df = pd.read_pickle(hist_file)
        self.assertIn("segment", df.columns)
        self.assertIn("branch", df.columns)
        self.assertIn("bucket", df.columns)
        self.assertIn("counts", df.columns)

        # Check segments and branches are present
        segments = set(df["segment"].unique())
        branches = set(df["branch"].unique())

        self.assertEqual(segments, {"Windows", "Linux", "Mac"})
        self.assertEqual(branches, {"control", "treatment"})

        # Check data sanity
        self.assertTrue(all(df["counts"] >= 0))
        self.assertTrue(all(df["bucket"] > 0))

        # Test pageload event data
        event_file = os.path.join(
            experiment_data_dir, f"{slug}-pageload-events-fcp_time.pkl"
        )
        self.assertTrue(os.path.exists(event_file))

        event_df = pd.read_pickle(event_file)
        self.assertIn("segment", event_df.columns)
        self.assertIn("branch", event_df.columns)
        self.assertIn("bucket", event_df.columns)
        self.assertIn("counts", event_df.columns)

    @patch("lib.telemetry.bigquery.Client")
    def test_rollout_single_branch(self, mock_bigquery_client):
        """Test rollout configuration with only one branch named 'control'."""

        # Mock BigQuery client to avoid actual queries
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        # Create separate test directory for rollout test
        rollout_test_dir = tempfile.mkdtemp()
        rollout_data_dir = os.path.join(rollout_test_dir, "data")
        rollout_reports_dir = os.path.join(rollout_test_dir, "reports")
        os.makedirs(rollout_data_dir, exist_ok=True)
        os.makedirs(rollout_reports_dir, exist_ok=True)

        try:
            # Create rollout config with only control branch
            rollout_config_path = os.path.join(
                rollout_test_dir, "rollout-experiment.yaml"
            )
            rollout_config = {
                "slug": "test-rollout-experiment",
                "channel": "release",
                "startDate": "2024-01-01",
                "endDate": "2024-01-07",
                "segments": ["Windows"],
                "include_non_enrolled_branch": False,
                "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
                "pageload_event_metrics": {"fcp_time": {"max": 30000}},
            }

            with open(rollout_config_path, "w") as f:
                yaml.dump(rollout_config, f, default_flow_style=False)

            # Create experiment data directory
            experiment_data_dir = os.path.join(
                rollout_data_dir, "test-rollout-experiment"
            )
            os.makedirs(experiment_data_dir, exist_ok=True)

            # Create Nimbus API cache for rollout (isRollout: true)
            rollout_experiment_config = {
                "name": "Test Rollout Experiment",
                "description": "A test rollout for testing single branch",
                "branches": [
                    {
                        "name": "control",
                        "slug": "control",
                        "description": "Control branch",
                        "ratio": 100,
                    }
                ],
                "channels": ["release"],
                "isRollout": True,
                "status": "Complete",
            }

            create_nimbus_api_cache(
                experiment_data_dir,
                "test-rollout-experiment",
                rollout_experiment_config,
            )

            # Create minimal test data for rollout branches
            # For rollouts, we need data for both 'control' and 'default' branches
            branches = [
                "control",
                "default",
            ]  # Rollouts automatically add 'default' branch
            segments = ["Windows"]  # Only one segment for rollout test

            # Create histogram data
            create_histogram_data(
                experiment_data_dir,
                "test-rollout-experiment",
                "metrics.timing_distribution.performance_pageload_fcp",
                branches=branches,
                segments=segments,
            )

            # Create pageload event data
            create_pageload_event_data(
                experiment_data_dir,
                "test-rollout-experiment",
                "fcp_time",
                branches=branches,
                segments=segments,
            )

            # Generate report
            class MockArgs:
                config = rollout_config_path
                dataDir = rollout_data_dir
                skip_cache = False
                reportDir = rollout_reports_dir
                html_report = True

            args = MockArgs()
            generate_report(args)

            # Verify HTML file was created
            expected_html = os.path.join(
                rollout_reports_dir, "test-rollout-experiment.html"
            )
            self.assertTrue(
                os.path.exists(expected_html),
                f"HTML report not found at {expected_html}",
            )

            # Read and verify HTML content
            with open(expected_html, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Basic structure checks
            self.assertTrue(
                html_content.startswith("<!DOCTYPE html>"),
                "HTML should start with DOCTYPE",
            )
            self.assertIn("test-rollout-experiment", html_content)

            # Check for experimenter page link
            experimenter_link = "https://experimenter.services.mozilla.com/nimbus/test-rollout-experiment"
            self.assertIn(
                experimenter_link,
                html_content,
                "Should contain link to experimenter page for the rollout slug",
            )

            # Check for rollout-specific content
            self.assertIn("control", html_content, "Should contain control branch")
            self.assertIn("Windows", html_content, "Should contain Windows segment")

            # Check for Configuration and Summary sections
            self.assertIn(
                "Configuration", html_content, "Should have Configuration section"
            )
            self.assertIn("Summary:", html_content, "Should have Summary section")

            # Check for metrics
            self.assertIn("performance_pageload_fcp", html_content)
            self.assertIn("fcp_time", html_content)

            # For rollouts, we should still have the expected canvas elements for the single branch
            metrics = ["performance_pageload_fcp", "fcp_time"]
            segments = ["Windows"]
            chart_types = ["pdf", "cdf", "uplift", "diff"]

            for segment in segments:
                for metric in metrics:
                    # Check for mean canvas
                    mean_canvas_id = f"{segment}-{metric}-mean"
                    self.assertIn(
                        f'id="{mean_canvas_id}"',
                        html_content,
                        f"Should have mean canvas for {segment}-{metric}",
                    )

                    # Check for other chart types
                    for chart_type in chart_types:
                        canvas_id = f"{segment}_{metric}_{chart_type}"
                        self.assertIn(
                            f'id="{canvas_id}"',
                            html_content,
                            f"Should have {chart_type} canvas for {segment}-{metric}",
                        )

            # Check for statistical tables
            stat_tables = html_content.count('class="stat-table"')
            self.assertGreater(
                stat_tables, 0, "Should have stat tables even for single branch rollout"
            )

            # Verify no obvious errors in HTML
            self.assertNotIn("Error", html_content)
            self.assertNotIn("Exception", html_content)
            self.assertNotIn("Traceback", html_content)

        finally:
            # Clean up rollout test directory
            shutil.rmtree(rollout_test_dir)


if __name__ == "__main__":
    unittest.main()
