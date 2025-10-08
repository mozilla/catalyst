#!/usr/bin/env python3

import unittest
import sys
import os
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path to import lib.telemetry
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.telemetry import (
    clean_sql_query,
    segments_are_all_OS,
    config_has_custom_branches,
    config_has_custom_segments,
    invalidDataSet,
    getInvalidBranchSegments,
    TelemetryClient,
)


class TestTelemetry(unittest.TestCase):
    """Tests for telemetry functions."""

    def test_clean_sql_query_basic(self):
        """Test clean_sql_query basic functionality."""
        query = "SELECT * FROM table\n\nWHERE condition = 'value'"
        result = clean_sql_query(query)
        self.assertIsInstance(result, str)
        self.assertIn("SELECT", result)
        self.assertIn("WHERE", result)

    def test_clean_sql_query_with_empty_lines(self):
        """Test clean_sql_query removes empty lines."""
        query = """
        SELECT * FROM table

        WHERE condition = 'value'

        ORDER BY column
        """
        result = clean_sql_query(query)

        # Should not contain double newlines
        self.assertNotIn("\n\n", result)
        self.assertIn("SELECT", result)
        self.assertIn("WHERE", result)
        self.assertIn("ORDER BY", result)

    def test_clean_sql_query_empty_string(self):
        """Test clean_sql_query with empty string."""
        result = clean_sql_query("")
        self.assertEqual(result, "")

    def test_clean_sql_query_only_whitespace(self):
        """Test clean_sql_query with only whitespace."""
        query = "\n\n   \n   \n\n"
        result = clean_sql_query(query)
        self.assertEqual(result, "")

    def test_segments_are_all_os_valid(self):
        """Test segments_are_all_OS with valid OS segments."""
        os_segments = ["Windows", "Linux", "Mac", "Android"]
        self.assertTrue(segments_are_all_OS(os_segments))

    def test_segments_are_all_os_invalid(self):
        """Test segments_are_all_OS with invalid segments."""
        mixed_segments = ["Windows", "Custom", "Linux"]
        self.assertFalse(segments_are_all_OS(mixed_segments))

    def test_segments_are_all_os_empty(self):
        """Test segments_are_all_OS with empty list."""
        self.assertTrue(segments_are_all_OS([]))

    def test_segments_are_all_os_case_sensitive(self):
        """Test segments_are_all_OS is case sensitive."""
        lowercase_segments = ["windows", "linux"]
        self.assertFalse(segments_are_all_OS(lowercase_segments))

    def test_segments_are_all_os_partial_match(self):
        """Test segments_are_all_OS with partial OS matches."""
        partial_segments = ["Windows_10", "Linux_Ubuntu"]
        self.assertFalse(segments_are_all_OS(partial_segments))

    def test_config_has_custom_branches_false(self):
        """Test config_has_custom_branches returns false for standard config."""
        config = {"slug": "test", "channel": "nightly"}
        self.assertFalse(config_has_custom_branches(config))

    def test_config_has_custom_branches_empty_branches(self):
        """Test config_has_custom_branches with empty branches."""
        config = {"branches": []}
        self.assertFalse(config_has_custom_branches(config))

    def test_config_has_custom_branches_with_branches(self):
        """Test config_has_custom_branches with actual branches."""
        config = {"has_custom_branches": True}
        self.assertTrue(config_has_custom_branches(config))

    def test_config_has_custom_segments_false(self):
        """Test config_has_custom_segments returns false for OS-only."""
        config = {"segments": ["Windows", "Linux", "Mac"]}
        self.assertFalse(config_has_custom_segments(config))

    def test_config_has_custom_segments_empty(self):
        """Test config_has_custom_segments with empty segments."""
        config = {"segments": []}
        self.assertFalse(config_has_custom_segments(config))

    def test_config_has_custom_segments_true(self):
        """Test config_has_custom_segments with custom segments."""
        config = {"custom_segments_info": {"segment_name": "custom"}}
        self.assertTrue(config_has_custom_segments(config))

    def test_invalid_data_set_with_sufficient_data(self):
        """Test invalidDataSet with sufficient sample sizes."""
        df = pd.DataFrame(
            {
                "segment": ["Windows", "Linux", "Windows", "Linux"],
                "branch": ["control", "control", "treatment", "treatment"],
                "counts": [2000, 1500, 1800, 1600],  # Above 1000 threshold
            }
        )
        branches = [{"name": "control"}, {"name": "treatment"}]
        segments = ["Windows", "Linux"]

        result = invalidDataSet(df, "test_metric", branches, segments)
        self.assertFalse(result)

    def test_invalid_data_set_with_insufficient_data(self):
        """Test invalidDataSet with insufficient sample sizes."""
        df = pd.DataFrame(
            {
                "segment": ["Windows", "Linux", "Windows", "Linux"],
                "branch": ["control", "control", "treatment", "treatment"],
                "counts": [100, 150, 120, 180],  # Below 1000 threshold
            }
        )
        branches = [{"name": "control"}, {"name": "treatment"}]
        segments = ["Windows", "Linux"]

        result = invalidDataSet(df, "test_metric", branches, segments)
        # invalidDataSet only returns True for completely empty data, not insufficient sample sizes
        self.assertFalse(result)

    def test_invalid_data_set_empty_dataframe(self):
        """Test invalidDataSet with empty DataFrame."""
        df = pd.DataFrame()
        branches = [{"name": "control"}]
        segments = ["Windows"]

        result = invalidDataSet(df, "test_metric", branches, segments)
        self.assertTrue(result)

    def test_get_invalid_branch_segments_with_valid_data(self):
        """Test getInvalidBranchSegments with all valid data."""
        df = pd.DataFrame(
            {
                "segment": ["Windows", "Linux", "Windows", "Linux"],
                "branch": ["control", "control", "treatment", "treatment"],
                "counts": [2000, 1500, 1800, 1200],
            }
        )
        branches = [{"name": "control"}, {"name": "treatment"}]
        segments = ["Windows", "Linux"]

        result = getInvalidBranchSegments(df, "test_metric", branches, segments)
        self.assertEqual(result, [])

    def test_get_invalid_branch_segments_with_missing_data(self):
        """Test getInvalidBranchSegments with missing combinations."""
        df = pd.DataFrame(
            {
                "segment": ["Windows", "Windows"],
                "branch": ["control", "treatment"],
                "counts": [2000, 1800],
            }
        )
        branches = [{"name": "control"}, {"name": "treatment"}]
        segments = ["Windows", "Linux"]

        result = getInvalidBranchSegments(df, "test_metric", branches, segments)
        expected = [("control", "Linux"), ("treatment", "Linux")]
        self.assertEqual(sorted(result), sorted(expected))

    def test_get_invalid_branch_segments_insufficient_counts(self):
        """Test getInvalidBranchSegments with insufficient counts."""
        df = pd.DataFrame(
            {
                "segment": ["Windows", "Linux"],
                "branch": ["control", "control"],
                "counts": [500, 800],  # Below 1000 threshold
            }
        )
        branches = [{"name": "control"}]
        segments = ["Windows", "Linux"]

        result = getInvalidBranchSegments(df, "test_metric", branches, segments)
        expected = [("control", "Windows"), ("control", "Linux")]
        self.assertEqual(sorted(result), sorted(expected))

    @patch("lib.telemetry.bigquery.Client")
    def test_telemetry_client_initialization(self, mock_bigquery_client):
        """Test TelemetryClient initialization."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        config = {
            "slug": "test-experiment",
            "histograms": [],
            "pageload_event_metrics": [],
            "crash_event_metrics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TelemetryClient(temp_dir, config, skipCache=False)

            # Check basic initialization
            self.assertEqual(client.config, config)
            self.assertEqual(client.dataDir, temp_dir)
            self.assertFalse(client.skipCache)
            self.assertEqual(client.max_workers, 4)  # default value
            self.assertIsInstance(client.queries, list)

    @patch("lib.telemetry.bigquery.Client")
    def test_telemetry_client_custom_max_workers(self, mock_bigquery_client):
        """Test TelemetryClient with custom max_workers."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        config = {
            "slug": "test-experiment",
            "histograms": [],
            "pageload_event_metrics": [],
            "crash_event_metrics": [],
            "max_parallel_queries": 8,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TelemetryClient(temp_dir, config, skipCache=True)

            # Check custom configuration
            self.assertEqual(client.max_workers, 8)
            self.assertTrue(client.skipCache)

    @patch("lib.telemetry.bigquery.Client")
    def test_telemetry_client_check_existing_data(self, mock_bigquery_client):
        """Test TelemetryClient checkForExistingData method."""
        mock_client = MagicMock()
        mock_bigquery_client.return_value = mock_client

        config = {
            "slug": "test-experiment",
            "histograms": [],
            "pageload_event_metrics": [],
            "crash_event_metrics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            client = TelemetryClient(temp_dir, config, skipCache=False)

            # Test with non-existent file
            result = client.checkForExistingData("non_existent_file.csv")
            self.assertIsNone(result)

            # Create a test pickle file
            test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            test_file_path = os.path.join(temp_dir, "test_data.pkl")
            test_data.to_pickle(test_file_path)

            # Test with existing file
            result = client.checkForExistingData(test_file_path)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 3)
            self.assertEqual(list(result.columns), ["col1", "col2"])

    @patch("lib.telemetry.bigquery.Client")
    def test_telemetry_client_authentication_check_success(self, mock_bigquery_client):
        """Test TelemetryClient authentication check with successful auth."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = MagicMock()
        mock_client.query.return_value = mock_job
        mock_bigquery_client.return_value = mock_client

        config = {
            "slug": "test-experiment",
            "histograms": [],
            "pageload_event_metrics": [],
            "crash_event_metrics": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # This should not raise an exception
            client = TelemetryClient(temp_dir, config, skipCache=False)
            self.assertIsNotNone(client)

    @patch("lib.telemetry.sys.exit")
    @patch("lib.telemetry.bigquery.Client")
    def test_telemetry_client_authentication_check_failure(self, mock_bigquery_client, mock_exit):
        """Test TelemetryClient authentication check with auth failure."""
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.side_effect = Exception("Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate.")
        mock_client.query.return_value = mock_job
        mock_bigquery_client.return_value = mock_client

        config = {
            "slug": "test-experiment",
            "histograms": {},
            "pageload_event_metrics": {},
            "crash_event_metrics": {},
            "max_parallel_queries": 4,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # This should call sys.exit(1)
            TelemetryClient(temp_dir, config, skipCache=False)
            mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
