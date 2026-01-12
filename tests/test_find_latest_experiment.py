#!/usr/bin/env python3
"""
Test suite for bin/find-latest-experiment script.

This test covers the experiment discovery, filtering, configuration generation,
and report generation orchestration functionality.
"""

import os
import sys
import tempfile
import shutil
import unittest
import json
import yaml
from datetime import datetime, timedelta

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the script functions by executing it
script_path = os.path.join(
    os.path.dirname(__file__), "..", "bin", "find-latest-experiment"
)

# Load the script content and extract functions
with open(script_path, "r") as f:
    script_content = f.read()

# Create a local namespace and execute the script
script_globals = {"__file__": script_path}
exec(compile(script_content, script_path, "exec"), script_globals)

# Extract the functions we want to test
is_supported_experiment = script_globals["is_supported_experiment"]
is_recent_experiment = script_globals["is_recent_experiment"]
filter_and_sort = script_globals["filter_and_sort"]
extract_existing_reports = script_globals["extract_existing_reports"]
generate_histogram_metrics = script_globals["generate_histogram_metrics"]
generate_event_metrics = script_globals["generate_event_metrics"]
create_config_for_experiment = script_globals["create_config_for_experiment"]
load_failures = script_globals["load_failures"]
append_failure = script_globals["append_failure"]


class TestFindLatestExperiment(unittest.TestCase):
    """Test the find-latest-experiment script functionality."""

    def setUp(self):
        """Create temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        self.index_file = os.path.join(self.test_dir, "index.html")
        self.failures_file = os.path.join(self.test_dir, "failures.json")

        # Create mock index.html file
        self.create_mock_index_file()

        # Sample experiment data
        self.sample_experiments = [
            {
                "slug": "test-experiment-1",
                "appName": "firefox_desktop",
                "endDate": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "isRollout": False,
                "branches": [
                    {"slug": "control", "ratio": 0.5},
                    {"slug": "treatment", "ratio": 0.5},
                ],
            },
            {
                "slug": "test-experiment-2",
                "appName": "fenix",
                "endDate": (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
                "isRollout": False,
                "branches": [
                    {"slug": "control", "ratio": 0.5},
                    {"slug": "treatment", "ratio": 0.5},
                ],
            },
        ]

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def create_mock_index_file(self):
        """Create a mock index.html file with existing reports."""
        index_content = """
        <html>
        <body>
            <table class="experiment-table">
                <tr>
                    <th>Experiment</th>
                    <th>Start Date</th>
                    <th>End Date</th>
                    <th>Channel</th>
                </tr>
                <tr>
                    <td>existing-experiment</td>
                    <td>2024-01-01</td>
                    <td>2024-01-07</td>
                    <td>release</td>
                </tr>
            </table>
        </body>
        </html>
        """
        with open(self.index_file, "w") as f:
            f.write(index_content)

    def test_is_supported_experiment(self):
        """Test experiment platform and branch validation."""
        # Test supported desktop experiment
        desktop_exp = {
            "appName": "firefox_desktop",
            "branches": [{"slug": "control"}, {"slug": "treatment"}],
            "isRollout": False,
        }
        self.assertTrue(is_supported_experiment(desktop_exp))

        # Test supported mobile experiment
        mobile_exp = {
            "appName": "fenix",
            "branches": [{"slug": "control"}, {"slug": "treatment"}],
            "isRollout": False,
        }
        self.assertTrue(is_supported_experiment(mobile_exp))

        # Test unsupported platform
        ios_exp = {
            "appName": "firefox_ios",
            "branches": [{"slug": "control"}],
            "isRollout": False,
        }
        self.assertFalse(is_supported_experiment(ios_exp))

        # Test experiment with no branches
        no_branches_exp = {
            "appName": "firefox_desktop",
            "branches": [],
            "isRollout": False,
        }
        self.assertFalse(is_supported_experiment(no_branches_exp))

        # Test rollout (should be rejected - we don't trust non-enrolled as control)
        rollout_exp = {
            "appName": "firefox_desktop",
            "branches": [{"slug": "rollout", "ratio": 1.0}],
            "isRollout": True,
        }
        self.assertFalse(is_supported_experiment(rollout_exp))

        # Test single branch experiment (should be rejected - no control)
        single_branch_exp = {
            "appName": "firefox_desktop",
            "branches": [{"slug": "treatment", "ratio": 0.5}],
            "isRollout": False,
        }
        self.assertFalse(is_supported_experiment(single_branch_exp))

    def test_is_recent_experiment(self):
        """Test experiment recency filtering."""
        # Test recent experiment (within 14 days)
        recent_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        self.assertTrue(is_recent_experiment(recent_date))

        # Test old experiment (older than 14 days)
        old_date = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
        self.assertFalse(is_recent_experiment(old_date))

        # Test custom days parameter
        week_old_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        self.assertTrue(is_recent_experiment(week_old_date, days=7))
        self.assertFalse(is_recent_experiment(week_old_date, days=3))

    def test_filter_and_sort(self):
        """Test experiment filtering and sorting by end date."""
        experiments = [
            {"slug": "exp1", "endDate": "2024-01-15"},
            {"slug": "exp2", "endDate": None},  # Should be filtered out
            {"slug": "exp3", "endDate": "2024-01-10"},
            {"slug": "exp4", "endDate": "2024-01-20"},
        ]

        filter_and_sort(experiments)

        # Should remove None endDate and sort by date
        expected_slugs = ["exp3", "exp1", "exp4"]
        actual_slugs = [exp["slug"] for exp in experiments]
        self.assertEqual(actual_slugs, expected_slugs)

    def test_extract_existing_reports(self):
        """Test extraction of existing reports from index.html."""
        reports = extract_existing_reports(self.index_file)

        self.assertIn("existing-experiment", reports)
        self.assertEqual(reports["existing-experiment"]["start_date"], "2024-01-01")
        self.assertEqual(reports["existing-experiment"]["end_date"], "2024-01-07")
        self.assertEqual(reports["existing-experiment"]["channel"], "release")

    def test_generate_histogram_metrics(self):
        """Test histogram metrics generation."""
        exp = {"slug": "test-exp"}
        histograms = generate_histogram_metrics(exp)

        # Should return default histograms (including labeled_counter with aggregate mode)
        expected_histograms = [
            "metrics.memory_distribution.memory_total",
            "metrics.timing_distribution.performance_pageload_fcp",
            "metrics.timing_distribution.performance_pageload_load_time",
            "metrics.timing_distribution.perf_largest_contentful_paint",
            {
                "metrics.labeled_counter.power_cpu_time_per_process_type_ms": {
                    "aggregate": "percentiles"
                }
            },
        ]
        self.assertEqual(histograms, expected_histograms)

    def test_generate_event_metrics(self):
        """Test event metrics generation."""
        exp = {"slug": "test-exp"}
        events = generate_event_metrics(exp)

        # Should return default events
        expected_events = {
            "fcp_time": {"max": 30000},
            "lcp_time": {"max": 30000},
            "load_time": {"max": 30000},
            "response_time": {"max": 30000},
        }
        self.assertEqual(events, expected_events)

    def test_create_config_for_experiment_desktop(self):
        """Test configuration creation for desktop experiments."""
        exp = {"slug": "desktop-test", "appName": "firefox_desktop"}

        # Change to test directory to avoid creating files in project root
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        try:
            args = create_config_for_experiment(exp)

            # Check args structure
            self.assertEqual(args.config, "desktop-test.yaml")
            self.assertEqual(args.dataDir, "data")
            self.assertEqual(args.reportDir, "reports")
            self.assertFalse(args.skip_cache)
            self.assertTrue(args.html_report)

            # Check that config file was created
            config_file = os.path.join(self.test_dir, "desktop-test.yaml")
            self.assertTrue(os.path.exists(config_file))

            # Check config contents
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["slug"], "desktop-test")
            self.assertEqual(config["segments"], ["Windows", "Linux", "Mac"])
            self.assertIn("histograms", config)
            self.assertIn("events", config)
            # Check that crash events are included
            self.assertIn("crash", config["events"])
            # Check that pageload events are included
            pageload_events = [
                event
                for event in config["events"]
                if isinstance(event, dict) and "pageload" in event
            ]
            self.assertEqual(len(pageload_events), 1)

        finally:
            os.chdir(original_cwd)

    def test_create_config_for_experiment_mobile(self):
        """Test configuration creation for mobile experiments."""
        exp = {"slug": "mobile-test", "appName": "fenix"}

        original_cwd = os.getcwd()
        os.chdir(self.test_dir)

        try:
            create_config_for_experiment(exp)

            # Check that config file was created
            config_file = os.path.join(self.test_dir, "mobile-test.yaml")
            self.assertTrue(os.path.exists(config_file))

            # Check config contents
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            self.assertEqual(config["slug"], "mobile-test")
            self.assertEqual(config["segments"], ["Android"])
            self.assertIn("histograms", config)
            self.assertIn("events", config)
            # Check that crash events are included
            self.assertIn("crash", config["events"])
            # Check that pageload events are included
            pageload_events = [
                event
                for event in config["events"]
                if isinstance(event, dict) and "pageload" in event
            ]
            self.assertEqual(len(pageload_events), 1)

        finally:
            os.chdir(original_cwd)

    def test_load_failures(self):
        """Test loading failure records."""
        # Test with non-existent file
        failures = load_failures("nonexistent.json")
        self.assertEqual(failures, {})

        # Test with existing file
        test_failures = {
            "failed-exp": {
                "error_message": "Test error",
                "timestamp": "2024-01-01T12:00:00",
            }
        }
        with open(self.failures_file, "w") as f:
            json.dump(test_failures, f)

        failures = load_failures(self.failures_file)
        self.assertEqual(failures, test_failures)

    def test_append_failure(self):
        """Test appending failure records."""
        config_file = os.path.join(self.test_dir, "test-config.json")
        config_data = {"test": "config"}

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        error = Exception("Test error message")
        append_failure(self.failures_file, "test-slug", error, config_file)

        # Check that failure was recorded
        with open(self.failures_file, "r") as f:
            failures = json.load(f)

        self.assertIn("test-slug", failures)
        self.assertEqual(failures["test-slug"]["error_message"], "Test error message")
        self.assertEqual(failures["test-slug"]["config"], config_data)
        self.assertIn("timestamp", failures["test-slug"])
        self.assertIn("traceback", failures["test-slug"])

    def test_function_coverage(self):
        """Test that all key functions are accessible and callable."""
        # Test that we can call all the main functions without errors
        test_exp = {
            "slug": "test",
            "appName": "firefox_desktop",
            "endDate": "2024-01-01",
            "isRollout": False,
            "branches": [{"slug": "control", "ratio": 0.5}],
        }

        # These should not raise exceptions
        self.assertIsNotNone(is_supported_experiment(test_exp))
        self.assertIsNotNone(is_recent_experiment("2024-01-01"))
        self.assertIsNotNone(generate_histogram_metrics(test_exp))
        self.assertIsNotNone(generate_event_metrics(test_exp))
        self.assertIsNotNone(load_failures("nonexistent.json"))

        print("✓ All key functions are accessible and working")


if __name__ == "__main__":
    unittest.main()
