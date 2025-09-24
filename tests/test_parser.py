#!/usr/bin/env python3

import unittest
import os
import json
import tempfile
import shutil
import sys
import requests
from unittest.mock import patch, mock_open, MagicMock

# Add the parent directory to sys.path to import lib.parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.parser import (
    checkForLocalFile,
    parseConfigFile,
    annotatePageloadEventMetrics,
    annotateHistograms,
    extractValuesFromAPI,
    retrieveNimbusAPI,
    parseNimbusAPI,
    parseEventsConfiguration,
    annotateMetrics,
)


class TestParser(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = os.path.dirname(__file__)
        self.fixtures_dir = os.path.join(self.test_dir, "fixtures")
        self.temp_dir = tempfile.mkdtemp()

        # Load mock data
        with open(os.path.join(self.fixtures_dir, "mock_probe_index.json")) as f:
            self.mock_probe_index = json.load(f)

        with open(os.path.join(self.fixtures_dir, "mock_nimbus_response.json")) as f:
            self.mock_nimbus_response = json.load(f)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_checkForLocalFile_exists(self):
        """Test checkForLocalFile with existing file"""
        test_file = os.path.join(self.fixtures_dir, "mock_probe_index.json")
        result = checkForLocalFile(test_file)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("glean", result)

    def test_checkForLocalFile_not_exists(self):
        """Test checkForLocalFile with non-existent file"""
        result = checkForLocalFile("/path/that/does/not/exist.json")
        self.assertIsNone(result)

    def test_checkForLocalFile_invalid_json(self):
        """Test checkForLocalFile with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content {")
            temp_file = f.name

        try:
            result = checkForLocalFile(temp_file)
            self.assertIsNone(result)
        finally:
            os.unlink(temp_file)

    def test_parseConfigFile_experiment(self):
        """Test parseConfigFile with experiment config"""
        config_file = os.path.join(self.fixtures_dir, "test_config.yaml")
        result = parseConfigFile(config_file)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["slug"], "test-experiment")
        self.assertTrue(result["is_experiment"])

        # After parsing, config should have new events format
        self.assertIn("events", result)

        # Call annotateMetrics to convert to internal format
        annotateMetrics(result)

        # Now should have legacy internal format
        self.assertIn("pageload_event_metrics", result)
        self.assertEqual(result["pageload_event_metrics"]["fcp_time"]["max"], 20000)

    def test_parseConfigFile_rollout(self):
        """Test parseConfigFile with rollout config (has branches)"""
        config_file = os.path.join(self.fixtures_dir, "test_config_with_branches.yaml")
        result = parseConfigFile(config_file)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["slug"], "test-rollout")
        self.assertFalse(result["is_experiment"])  # Should be False when branches exist
        self.assertIn("branches", result)

    def test_parseConfigFile_invalid_path(self):
        """Test parseConfigFile with invalid file path"""
        with self.assertRaises(FileNotFoundError):
            parseConfigFile("/path/that/does/not/exist.yaml")

    @patch("lib.parser.loadProbeIndex")
    def test_annotatePageloadEventMetrics_valid(self, mock_load_probe_index):
        """Test annotatePageloadEventMetrics with valid config"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {
            "pageload_event_metrics": {
                "fcp_time": {"max": 20000},
                "lcp_time": {"max": 30000},
            }
        }

        annotatePageloadEventMetrics(config, self.mock_probe_index)

        # Check that metrics were annotated correctly
        self.assertIn("desc", config["pageload_event_metrics"]["fcp_time"])
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["min"], 0)
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["max"], 20000)
        self.assertEqual(config["pageload_event_metrics"]["lcp_time"]["min"], 0)
        self.assertEqual(config["pageload_event_metrics"]["lcp_time"]["max"], 30000)

    def test_annotatePageloadEventMetrics_invalid_format(self):
        """Test annotatePageloadEventMetrics with invalid old format"""
        config = {
            "pageload_event_metrics": {
                "fcp_time": [0, 20000]  # Old format should be rejected
            }
        }

        with self.assertRaises(SystemExit):
            annotatePageloadEventMetrics(config, self.mock_probe_index)

    def test_annotatePageloadEventMetrics_missing_max(self):
        """Test annotatePageloadEventMetrics with missing max parameter"""
        config = {
            "pageload_event_metrics": {
                "fcp_time": {}  # No max parameter, should use default
            }
        }

        annotatePageloadEventMetrics(config, self.mock_probe_index)

        # Should use default max value of 30000
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["max"], 30000)

    def test_annotatePageloadEventMetrics_unknown_metric(self):
        """Test annotatePageloadEventMetrics with unknown metric"""
        config = {"pageload_event_metrics": {"unknown_metric": {"max": 20000}}}

        with self.assertRaises(SystemExit):
            annotatePageloadEventMetrics(config, self.mock_probe_index)

    @patch("lib.parser.loadProbeIndex")
    def test_annotateHistograms_glean_metric(self, mock_load_probe_index):
        """Test annotateHistograms with Glean metric"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"]
        }

        annotateHistograms(config, self.mock_probe_index)

        metric_name = "metrics.timing_distribution.performance_pageload_fcp"
        self.assertIn(metric_name, config["histograms"])
        self.assertTrue(config["histograms"][metric_name]["glean"])
        self.assertEqual(config["histograms"][metric_name]["kind"], "numerical")

    @patch("lib.parser.loadProbeIndex")
    def test_annotateHistograms_legacy_metric(self, mock_load_probe_index):
        """Test annotateHistograms with legacy metric"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"histograms": ["payload.histograms.memory_total"]}

        annotateHistograms(config, self.mock_probe_index)

        metric_name = "payload.histograms.memory_total"
        self.assertIn(metric_name, config["histograms"])
        self.assertFalse(config["histograms"][metric_name]["glean"])
        self.assertEqual(config["histograms"][metric_name]["kind"], "numerical")

    @patch("lib.parser.loadProbeIndex")
    def test_annotateHistograms_unknown_glean_metric(self, mock_load_probe_index):
        """Test annotateHistograms with unknown Glean metric not in probe index"""
        mock_load_probe_index.return_value = self.mock_probe_index
        config = {
            "histograms": ["metrics.timing_distribution.fake_metric_not_in_probe_index"]
        }

        with self.assertRaises(SystemExit):
            annotateHistograms(config, self.mock_probe_index)

    @patch("lib.parser.loadProbeIndex")
    def test_annotateHistograms_unknown_legacy_metric(self, mock_load_probe_index):
        """Test annotateHistograms with unknown legacy metric not in probe index"""
        mock_load_probe_index.return_value = self.mock_probe_index
        config = {"histograms": ["payload.histograms.bogus_legacy_metric"]}

        with self.assertRaises(SystemExit):
            annotateHistograms(config, self.mock_probe_index)

    @patch("lib.parser.loadProbeIndex")
    def test_annotateHistograms_completely_invalid_metric(self, mock_load_probe_index):
        """Test annotateHistograms with completely invalid metric format"""
        mock_load_probe_index.return_value = self.mock_probe_index
        config = {"histograms": ["totally.invalid.metric.path.not_anywhere"]}

        with self.assertRaises(SystemExit):
            annotateHistograms(config, self.mock_probe_index)

    def test_extractValuesFromAPI_basic(self):
        """Test extractValuesFromAPI with basic response"""
        result = extractValuesFromAPI(self.mock_nimbus_response)

        self.assertEqual(result["startDate"], "2024-01-01")
        self.assertEqual(result["endDate"], "2024-01-31")
        self.assertEqual(result["channel"], "release")
        self.assertFalse(result["isRollout"])
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual(result["branches"][0]["name"], "control")
        self.assertEqual(result["branches"][1]["name"], "treatment")

    def test_extractValuesFromAPI_no_end_date(self):
        """Test extractValuesFromAPI when endDate is None"""
        api_response = self.mock_nimbus_response.copy()
        api_response["endDate"] = None

        result = extractValuesFromAPI(api_response)

        # Should set endDate to current date
        self.assertIsNotNone(result["endDate"])
        # Should be in YYYY-MM-DD format
        self.assertRegex(result["endDate"], r"\d{4}-\d{2}-\d{2}")

    def test_extractValuesFromAPI_channel_priority(self):
        """Test extractValuesFromAPI channel priority selection"""
        # Test beta priority over nightly
        api_response = self.mock_nimbus_response.copy()
        api_response["channels"] = ["nightly", "beta"]
        result = extractValuesFromAPI(api_response)
        self.assertEqual(result["channel"], "beta")

        # Test release priority over beta
        api_response["channels"] = ["beta", "release"]
        result = extractValuesFromAPI(api_response)
        self.assertEqual(result["channel"], "release")

    def test_extractValuesFromAPI_unsupported_channel(self):
        """Test extractValuesFromAPI with unsupported channels"""
        api_response = self.mock_nimbus_response.copy()
        api_response["channels"] = ["unsupported"]

        with self.assertRaises(ValueError) as context:
            extractValuesFromAPI(api_response)

        self.assertIn("No supported channel found", str(context.exception))

    @patch("requests.get")
    @patch("lib.parser.checkForLocalFile")
    def test_retrieveNimbusAPI_cached(self, mock_check_file, mock_get):
        """Test retrieveNimbusAPI with cached data"""
        mock_check_file.return_value = self.mock_nimbus_response

        result = retrieveNimbusAPI(self.temp_dir, "test-experiment", False)

        self.assertEqual(result, self.mock_nimbus_response)
        mock_get.assert_not_called()  # Should not make HTTP request

    @patch("requests.get")
    @patch("lib.parser.checkForLocalFile")
    @patch("builtins.open", new_callable=mock_open)
    def test_retrieveNimbusAPI_http_success(self, mock_file, mock_check_file, mock_get):
        """Test retrieveNimbusAPI with successful HTTP request"""
        mock_check_file.return_value = None  # No cached data
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = self.mock_nimbus_response
        mock_get.return_value = mock_response

        result = retrieveNimbusAPI(self.temp_dir, "test-experiment", False)

        self.assertEqual(result, self.mock_nimbus_response)
        mock_get.assert_called_once()
        mock_file.assert_called()  # Should save to cache

    @patch("requests.get")
    @patch("lib.parser.checkForLocalFile")
    def test_retrieveNimbusAPI_http_timeout(self, mock_check_file, mock_get):
        """Test retrieveNimbusAPI with HTTP timeout"""
        mock_check_file.return_value = None
        mock_get.side_effect = requests.exceptions.Timeout()

        with self.assertRaises(SystemExit):
            retrieveNimbusAPI(self.temp_dir, "test-experiment", False)

    @patch("requests.get")
    @patch("lib.parser.checkForLocalFile")
    def test_retrieveNimbusAPI_http_error(self, mock_check_file, mock_get):
        """Test retrieveNimbusAPI with HTTP error"""
        mock_check_file.return_value = None
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_get.return_value = mock_response

        with self.assertRaises(SystemExit):
            retrieveNimbusAPI(self.temp_dir, "test-experiment", False)

    @patch("lib.parser.retrieveNimbusAPI")
    @patch("lib.parser.extractValuesFromAPI")
    def test_parseNimbusAPI(self, mock_extract, mock_retrieve):
        """Test parseNimbusAPI function"""
        mock_retrieve.return_value = self.mock_nimbus_response
        mock_extract.return_value = {"parsed": "data"}

        result = parseNimbusAPI(self.temp_dir, "test-experiment", False)

        mock_retrieve.assert_called_once_with(self.temp_dir, "test-experiment", False)
        mock_extract.assert_called_once_with(self.mock_nimbus_response)
        self.assertEqual(result, {"parsed": "data"})

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_crash_only(self, mock_load_probe_index):
        """Test parseEventsConfiguration with crash events only"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": ["crash"]}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create crash_event_metrics
        self.assertIn("crash_event_metrics", config)
        self.assertIn("total_crashes", config["crash_event_metrics"])
        self.assertEqual(len(config["crash_event_metrics"]), 1)

        # Should create empty pageload_event_metrics
        self.assertIn("pageload_event_metrics", config)
        self.assertEqual(len(config["pageload_event_metrics"]), 0)

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_pageload_only_default(
        self, mock_load_probe_index
    ):
        """Test parseEventsConfiguration with pageload events using defaults"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": ["pageload"]}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create pageload_event_metrics with defaults
        self.assertIn("pageload_event_metrics", config)
        expected_metrics = ["fcp_time", "lcp_time", "load_time"]
        for metric in expected_metrics:
            self.assertIn(metric, config["pageload_event_metrics"])
            self.assertEqual(config["pageload_event_metrics"][metric]["max"], 30000)

        # Should create empty crash_event_metrics
        self.assertIn("crash_event_metrics", config)
        self.assertEqual(len(config["crash_event_metrics"]), 0)

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_pageload_custom(self, mock_load_probe_index):
        """Test parseEventsConfiguration with custom pageload metrics"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {
            "events": [
                {"pageload": {"fcp_time": {"max": 25000}, "lcp_time": {"max": 15000}}}
            ]
        }

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create pageload_event_metrics with custom values
        self.assertIn("pageload_event_metrics", config)
        self.assertIn("fcp_time", config["pageload_event_metrics"])
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["max"], 25000)
        self.assertIn("lcp_time", config["pageload_event_metrics"])
        self.assertEqual(config["pageload_event_metrics"]["lcp_time"]["max"], 15000)

        # Should not include default metrics that weren't specified
        self.assertNotIn("load_time", config["pageload_event_metrics"])

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_pageload_empty_config(
        self, mock_load_probe_index
    ):
        """Test parseEventsConfiguration with empty pageload config"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": [{"pageload": {}}]}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should use defaults when pageload config is empty
        self.assertIn("pageload_event_metrics", config)
        expected_metrics = ["fcp_time", "lcp_time", "load_time"]
        for metric in expected_metrics:
            self.assertIn(metric, config["pageload_event_metrics"])
            self.assertEqual(config["pageload_event_metrics"][metric]["max"], 30000)

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_both_events(self, mock_load_probe_index):
        """Test parseEventsConfiguration with both crash and pageload events"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": ["crash", {"pageload": {"fcp_time": {"max": 20000}}}]}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create both event types
        self.assertIn("crash_event_metrics", config)
        self.assertIn("total_crashes", config["crash_event_metrics"])

        self.assertIn("pageload_event_metrics", config)
        self.assertIn("fcp_time", config["pageload_event_metrics"])
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["max"], 20000)

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_empty_events(self, mock_load_probe_index):
        """Test parseEventsConfiguration with empty events list"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": []}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create empty metrics
        self.assertIn("pageload_event_metrics", config)
        self.assertEqual(len(config["pageload_event_metrics"]), 0)
        self.assertIn("crash_event_metrics", config)
        self.assertEqual(len(config["crash_event_metrics"]), 0)

    @patch("lib.parser.loadProbeIndex")
    def test_parseEventsConfiguration_no_events_key(self, mock_load_probe_index):
        """Test parseEventsConfiguration when events key is missing"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {}

        parseEventsConfiguration(config, self.mock_probe_index)

        # Should create empty metrics
        self.assertIn("pageload_event_metrics", config)
        self.assertEqual(len(config["pageload_event_metrics"]), 0)
        self.assertIn("crash_event_metrics", config)
        self.assertEqual(len(config["crash_event_metrics"]), 0)

    @patch("lib.parser.loadProbeIndex")
    def test_annotateMetrics_new_events_format(self, mock_load_probe_index):
        """Test annotateMetrics with new events format"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": ["crash", {"pageload": {"fcp_time": {"max": 25000}}}],
        }

        annotateMetrics(config)

        # Should process histograms
        self.assertIn("histograms", config)
        self.assertIn(
            "metrics.timing_distribution.performance_pageload_fcp", config["histograms"]
        )

        # Should process events and create legacy format
        self.assertIn("pageload_event_metrics", config)
        self.assertIn("fcp_time", config["pageload_event_metrics"])
        self.assertEqual(config["pageload_event_metrics"]["fcp_time"]["max"], 25000)

        self.assertIn("crash_event_metrics", config)
        self.assertIn("total_crashes", config["crash_event_metrics"])

    @patch("lib.parser.loadProbeIndex")
    def test_annotateMetrics_legacy_format_compatible(self, mock_load_probe_index):
        """Test annotateMetrics with mixed formats works in consolidated approach"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {
            "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
            "events": [{"pageload": {"fcp_time": {"max": 20000}}}],
        }

        # Should work fine with consolidated approach
        annotateMetrics(config)

        # Check that metrics stay in their original sections (no transformation at parse time)
        self.assertIn(
            "metrics.timing_distribution.performance_pageload_fcp", config["histograms"]
        )
        self.assertIn("fcp_time", config["pageload_event_metrics"])
        # Data type consolidation happens at analysis stage, not parsing stage

    @patch("lib.parser.loadProbeIndex")
    def test_annotateMetrics_crash_legacy_format_compatible(
        self, mock_load_probe_index
    ):
        """Test annotateMetrics with crash formats works in consolidated approach"""
        mock_load_probe_index.return_value = self.mock_probe_index

        config = {"events": ["crash"]}

        # Should work fine with consolidated approach
        annotateMetrics(config)

        # Check that crash metrics stay in their original section (no transformation at parse time)
        self.assertIn("total_crashes", config["crash_event_metrics"])
        # Data type consolidation happens at analysis stage, not parsing stage


if __name__ == "__main__":
    unittest.main()
