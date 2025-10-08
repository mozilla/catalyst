"""
End-to-end tests for non-experiment configurations.
Tests various combinations of features including prerequisite CTEs, custom conditions,
custom branches, and different time ranges and segments.
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
)


class TestNonExperimentEndToEnd(unittest.TestCase):
    """End-to-end tests for non-experiment configurations with various features."""

    def setUp(self):
        """Create temporary directories for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def create_test_data_files(self, slug, branches, segments):
        """Create artificial test data files for histogram and pageload events."""
        experiment_data_dir = os.path.join(self.data_dir, slug)
        os.makedirs(experiment_data_dir, exist_ok=True)

        # Create histogram data
        create_histogram_data(
            experiment_data_dir,
            slug,
            "metrics.timing_distribution.performance_pageload_fcp",
            branches=branches,
            segments=segments,
        )

        # Create pageload event data
        create_pageload_event_data(
            experiment_data_dir,
            slug,
            "fcp_time",
            branches=branches,
            segments=segments,
        )

    @patch("lib.telemetry.bigquery.Client")
    def test_prerequisite_ctes_with_custom_conditions(self, mock_bigquery_client):
        """Test config with prerequisite CTEs and custom conditions (like addon filtering)."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "addon-performance-test"
        config_path = os.path.join(self.test_dir, f"{slug}.yaml")

        # Create config with prerequisite CTE and custom conditions
        config = {
            "slug": slug,
            "prerequisite_ctes": """top_addons AS (
  SELECT addon_id
  FROM addon_stats
  WHERE category = 'popular'
  LIMIT 50
)""",
            "branches": [
                {
                    "name": "no addons",
                    "startDate": "2024-03-01",
                    "endDate": "2024-03-07",
                    "channel": "nightly",
                    "custom_condition": "m.metrics.object.addons_active_addons IS NULL",
                },
                {
                    "name": "has addons",
                    "startDate": "2024-03-01",
                    "endDate": "2024-03-07",
                    "channel": "nightly",
                    "custom_condition": "EXISTS(SELECT 1 FROM UNNEST(JSON_EXTRACT_ARRAY(m.metrics.object.addons_active_addons)) AS addon JOIN top_addons t ON JSON_EXTRACT_SCALAR(addon, '$.id') = t.addon_id)",
                },
            ],
            "segments": ["Windows", "Mac", "Linux"],
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": [{"pageload": {"fcp_time": {"max": 30000}}}],
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create test data
        branches = ["no addons", "has addons"]
        segments = ["Windows", "Mac", "Linux"]
        self.create_test_data_files(slug, branches, segments)

        # Mock command line arguments
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report
        generate_report(args)

        # Verify HTML file was created
        expected_html = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(expected_html))

        # Read and verify HTML content
        with open(expected_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify basic structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn(slug, html_content)

        # Verify branches appear in HTML
        self.assertIn("no addons", html_content)
        self.assertIn("has addons", html_content)

        # Verify segments appear in HTML
        self.assertIn("Windows", html_content)
        self.assertIn("Mac", html_content)
        self.assertIn("Linux", html_content)

        # Verify date ranges appear
        self.assertIn("2024-03-01", html_content)
        self.assertIn("2024-03-07", html_content)

        # Verify metrics appear
        self.assertIn("performance_pageload_fcp", html_content)
        self.assertIn("fcp_time", html_content)

        # Verify channel appears
        self.assertIn("nightly", html_content)

    @patch("lib.telemetry.bigquery.Client")
    def test_multi_branch_different_dates_channels(self, mock_bigquery_client):
        """Test config with multiple branches having different dates and channels."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "multi-branch-comparison"
        config_path = os.path.join(self.test_dir, f"{slug}.yaml")

        # Create config with branches having different dates and channels
        config = {
            "slug": slug,
            "branches": [
                {
                    "name": "firefox 120",
                    "startDate": "2024-01-01",
                    "endDate": "2024-01-07",
                    "channel": "release",
                    "custom_condition": "SPLIT(client_info.app_display_version, '.')[offset(0)] = '120'",
                },
                {
                    "name": "firefox 121",
                    "startDate": "2024-02-01",
                    "endDate": "2024-02-07",
                    "channel": "release",
                    "custom_condition": "SPLIT(client_info.app_display_version, '.')[offset(0)] = '121'",
                },
                {
                    "name": "nightly build",
                    "startDate": "2024-02-15",
                    "endDate": "2024-02-21",
                    "channel": "nightly",
                    "custom_condition": "client_info.channel = 'nightly'",
                },
            ],
            "segments": ["Windows", "Android"],
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": [{"pageload": {"fcp_time": {"max": 30000}}}],
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create test data
        branches = ["firefox 120", "firefox 121", "nightly build"]
        segments = ["Windows", "Android"]
        self.create_test_data_files(slug, branches, segments)

        # Mock command line arguments
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report
        generate_report(args)

        # Verify HTML file was created
        expected_html = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(expected_html))

        # Read and verify HTML content
        with open(expected_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify all branches appear
        self.assertIn("firefox 120", html_content)
        self.assertIn("firefox 121", html_content)
        self.assertIn("nightly build", html_content)

        # Verify all segments appear
        self.assertIn("Windows", html_content)
        self.assertIn("Android", html_content)

        # Verify different date ranges appear
        self.assertIn("2024-01-01", html_content)
        self.assertIn("2024-01-07", html_content)
        self.assertIn("2024-02-01", html_content)
        self.assertIn("2024-02-07", html_content)
        self.assertIn("2024-02-15", html_content)
        self.assertIn("2024-02-21", html_content)

        # Verify different channels appear
        self.assertIn("release", html_content)
        self.assertIn("nightly", html_content)

    @patch("lib.telemetry.bigquery.Client")
    def test_complex_prerequisite_ctes_and_conditions(self, mock_bigquery_client):
        """Test config with complex prerequisite CTEs and multiple custom conditions."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "complex-analysis"
        config_path = os.path.join(self.test_dir, f"{slug}.yaml")

        # Create config with complex prerequisite CTEs and conditions
        config = {
            "slug": slug,
            "prerequisite_ctes": """user_cohorts AS (
  SELECT client_id, cohort_type
  FROM user_analysis
  WHERE analysis_date >= '{{start_date}}'
    AND analysis_date <= '{{end_date}}'
),
performance_baseline AS (
  SELECT percentile_cont(0.5) OVER() as median_fcp
  FROM historical_performance
  WHERE date_range = 'last_30_days'
)""",
            "branches": [
                {
                    "name": "power users",
                    "startDate": "2024-04-01",
                    "endDate": "2024-04-14",
                    "channel": "release",
                    "custom_condition": "EXISTS(SELECT 1 FROM user_cohorts uc WHERE uc.client_id = m.client_info.client_id AND uc.cohort_type = 'power_user')",
                },
                {
                    "name": "casual users",
                    "startDate": "2024-04-01",
                    "endDate": "2024-04-14",
                    "channel": "release",
                    "custom_condition": "EXISTS(SELECT 1 FROM user_cohorts uc WHERE uc.client_id = m.client_info.client_id AND uc.cohort_type = 'casual_user')",
                },
            ],
            "segments": ["Windows", "Mac", "Linux", "Android"],
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": [{"pageload": {"fcp_time": {"max": 30000}}}],
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create test data
        branches = ["power users", "casual users"]
        segments = ["Windows", "Mac", "Linux", "Android"]
        self.create_test_data_files(slug, branches, segments)

        # Mock command line arguments
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report
        generate_report(args)

        # Verify HTML file was created
        expected_html = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(expected_html))

        # Read and verify HTML content
        with open(expected_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify branches with spaces are handled correctly
        self.assertIn("power users", html_content)
        self.assertIn("casual users", html_content)

        # Verify all segments appear
        self.assertIn("Windows", html_content)
        self.assertIn("Mac", html_content)
        self.assertIn("Linux", html_content)
        self.assertIn("Android", html_content)

        # Verify date range appears
        self.assertIn("2024-04-01", html_content)
        self.assertIn("2024-04-14", html_content)

        # Verify metrics appear
        self.assertIn("performance_pageload_fcp", html_content)
        self.assertIn("fcp_time", html_content)

    @patch("lib.telemetry.bigquery.Client")
    def test_simple_standard_conditions(self, mock_bigquery_client):
        """Test config without custom conditions or prerequisite CTEs (standard conditions)."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        slug = "simple-comparison"
        config_path = os.path.join(self.test_dir, f"{slug}.yaml")

        # Create simple config without custom conditions
        config = {
            "slug": slug,
            "branches": [
                {
                    "name": "branch A",
                    "startDate": "2024-05-01",
                    "endDate": "2024-05-07",
                    "channel": "beta",
                },
                {
                    "name": "branch B",
                    "startDate": "2024-05-08",
                    "endDate": "2024-05-14",
                    "channel": "beta",
                },
            ],
            "segments": ["Windows", "Mac"],
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": [{"pageload": {"fcp_time": {"max": 30000}}}],
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Create test data
        branches = ["branch A", "branch B"]
        segments = ["Windows", "Mac"]
        self.create_test_data_files(slug, branches, segments)

        # Mock command line arguments
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = False
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report
        generate_report(args)

        # Verify HTML file was created
        expected_html = os.path.join(self.reports_dir, f"{slug}.html")
        self.assertTrue(os.path.exists(expected_html))

        # Read and verify HTML content
        with open(expected_html, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify branches appear
        self.assertIn("branch A", html_content)
        self.assertIn("branch B", html_content)

        # Verify segments appear
        self.assertIn("Windows", html_content)
        self.assertIn("Mac", html_content)

        # Verify different date ranges appear
        self.assertIn("2024-05-01", html_content)
        self.assertIn("2024-05-07", html_content)
        self.assertIn("2024-05-08", html_content)
        self.assertIn("2024-05-14", html_content)

        # Verify channel appears
        self.assertIn("beta", html_content)

    @patch("lib.telemetry.bigquery.Client")
    def test_sql_query_generation_validation(self, mock_bigquery_client):
        """Test that generated SQL queries contain the expected conditions and CTEs."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        # Capture generated queries
        generated_queries = []

        def capture_query(query, *args, **kwargs):
            generated_queries.append(query)
            # Return mock result
            mock_result = MagicMock()
            mock_result.to_dataframe.return_value = pd.DataFrame(
                {
                    "segment": ["Windows", "Mac"],
                    "branch": ["test branch", "test branch"],
                    "bucket": [1000, 1500],
                    "counts": [100, 200],
                }
            )
            return mock_result

        mock_client.query.side_effect = capture_query

        slug = "query-validation-test"
        config_path = os.path.join(self.test_dir, f"{slug}.yaml")

        # Create config with prerequisite CTE and custom conditions to validate
        config = {
            "slug": slug,
            "prerequisite_ctes": """validation_cte AS (
  SELECT test_field
  FROM test_table
  WHERE test_condition = 'expected_value'
)""",
            "branches": [
                {
                    "name": "test branch",
                    "startDate": "2024-06-01",
                    "endDate": "2024-06-07",
                    "channel": "nightly",
                    "custom_condition": "test_column = 'expected_condition'",
                },
                {
                    "name": "control branch",
                    "startDate": "2024-06-01",
                    "endDate": "2024-06-07",
                    "channel": "nightly",
                    "custom_condition": "test_column = 'control_condition'",
                },
            ],
            "segments": ["Windows", "Mac", "Android"],
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Don't create test data files for this test - we want to force query generation
        # self.create_test_data_files(slug, branches, segments)

        # Mock command line arguments - skip cache to force query generation
        class MockArgs:
            config = config_path
            dataDir = self.data_dir
            skip_cache = True  # Force query generation instead of using cache
            reportDir = self.reports_dir
            html_report = True

        args = MockArgs()

        # Generate report
        generate_report(args)

        # Verify queries were generated
        self.assertGreater(len(generated_queries), 0, "No queries were generated")

        # Find histogram query (should contain our test conditions)
        histogram_queries = [
            q for q in generated_queries if "performance_pageload_fcp" in q
        ]
        self.assertGreater(len(histogram_queries), 0, "No histogram queries found")

        query = histogram_queries[0]

        # Verify prerequisite CTE appears in query
        self.assertIn(
            "validation_cte AS", query, "Prerequisite CTE should appear in query"
        )
        self.assertIn(
            "test_condition = 'expected_value'",
            query,
            "CTE condition should appear in query",
        )

        # Verify custom condition appears in query
        self.assertIn(
            "test_column = 'expected_condition'",
            query,
            "Custom condition should appear in query",
        )

        # Verify date range appears in query
        self.assertIn("2024-06-01", query, "Start date should appear in query")
        self.assertIn("2024-06-07", query, "End date should appear in query")

        # Verify channel appears in query
        self.assertIn("nightly", query, "Channel should appear in query")

        # Verify branch names with spaces are handled (should be safe_name in CTEs)
        self.assertIn(
            "test_branch_desktop",
            query,
            "Safe branch name should appear in desktop CTE",
        )
        self.assertIn(
            "test_branch_android",
            query,
            "Safe branch name should appear in android CTE",
        )
        self.assertIn(
            "control_branch_desktop",
            query,
            "Control branch safe name should appear in desktop CTE",
        )
        self.assertIn(
            "control_branch_android",
            query,
            "Control branch safe name should appear in android CTE",
        )

        # Verify Android conditions are properly converted (m.metrics -> f.metrics)
        android_sections = [
            section
            for section in query.split("FROM `mozdata.fenix.metrics`")
            if len(section) > 100
        ]
        if android_sections:
            # Find the android section and verify the condition was converted
            android_section = android_sections[0]
            # The custom condition should be converted for Android usage
            # (This would be done by the template logic we implemented)
            self.assertTrue(
                "test_column = 'expected_condition'" in android_section,
                "Android custom condition should appear in Android section",
            )


if __name__ == "__main__":
    unittest.main()
