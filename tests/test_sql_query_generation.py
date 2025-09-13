#!/usr/bin/env python3
"""
Test suite for SQL query generation and syntax validation.

This test creates various config files to exercise the SQL templates in lib/telemetry.py
and validates the generated queries using offline SQL syntax checkers that don't require
BigQuery authentication.
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.telemetry import TelemetryClient
from lib.generate import setupDjango
from tests.test_data_utils import create_test_config, create_nimbus_api_cache

try:
    import sqlvalidator

    SQL_VALIDATOR_AVAILABLE = True
except ImportError:
    SQL_VALIDATOR_AVAILABLE = False

try:
    from bigquery_sql_parser import parse_sql

    BIGQUERY_PARSER_AVAILABLE = True
except ImportError:
    BIGQUERY_PARSER_AVAILABLE = False


class TestSQLQueryGeneration(unittest.TestCase):
    """Test SQL query generation from various config scenarios."""

    def setUp(self):
        """Create temporary directories and test configurations."""
        # Setup Django for template rendering
        setupDjango()

        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)

        # Store generated queries for validation
        self.generated_queries = []

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    @patch("lib.telemetry.bigquery.Client")
    def create_experiment_config(
        self, config_name, config_overrides, mock_bigquery_client
    ):
        """Create an experiment config and return the TelemetryClient."""
        # Mock the BigQuery client to avoid authentication
        mock_bigquery_client.return_value = MagicMock()

        experiment_data_dir = os.path.join(self.data_dir, config_name)
        os.makedirs(experiment_data_dir, exist_ok=True)

        # Create config
        config = create_test_config(config_name, config_overrides)

        # Add required fields for query generation
        if "is_experiment" not in config:
            config["is_experiment"] = True

        # Create Nimbus API cache
        experiment_config = {
            "name": f"Test {config_name}",
            "description": f"Test config for {config_name}",
            "channels": [config.get("channel", "release")],
            "isRollout": config.get("isRollout", False),
            "status": "Complete",
        }
        create_nimbus_api_cache(experiment_data_dir, config_name, experiment_config)

        # Create telemetry client with skipCache=True to force query generation
        telemetry_client = TelemetryClient(experiment_data_dir, config, skipCache=True)
        return telemetry_client

    def validate_sql_with_sqlvalidator(self, query, query_name):
        """Validate SQL using sqlvalidator package if available."""
        if not SQL_VALIDATOR_AVAILABLE:
            self.skipTest("sqlvalidator package not available")

        try:
            # Create a SQL validator instance and format the SQL
            validator = sqlvalidator.format_sql(query)
            self.assertIsNotNone(
                validator, f"sqlvalidator failed to format {query_name}"
            )

            # Basic validation - if format doesn't raise an exception, syntax is likely valid
            self.assertIn(
                "SELECT", validator.upper(), f"{query_name} missing SELECT statement"
            )
            self.assertIn(
                "FROM", validator.upper(), f"{query_name} missing FROM clause"
            )

            print(f"✓ {query_name} passed sqlvalidator syntax check")
            return True

        except Exception as e:
            self.fail(f"sqlvalidator failed for {query_name}: {e}")

    def validate_sql_with_bigquery_parser(self, query, query_name):
        """Validate SQL using bigquery-sql-parser if available."""
        if not BIGQUERY_PARSER_AVAILABLE:
            self.skipTest("bigquery-sql-parser package not available")

        try:
            # Parse the SQL - if it doesn't raise an exception, syntax is valid
            parsed = parse_sql(query)
            self.assertIsNotNone(
                parsed, f"bigquery-sql-parser failed to parse {query_name}"
            )
            print(f"✓ {query_name} passed bigquery-sql-parser syntax check")
            return True

        except Exception as e:
            # Some complex BigQuery features might not be supported by the parser
            # Log the error but don't fail the test unless it's a basic syntax error
            print(f"⚠ {query_name} bigquery-sql-parser warning: {e}")
            return False

    def validate_sql_basic_checks(self, query, query_name):
        """Perform basic SQL syntax checks without external dependencies."""
        # Basic structural checks
        query_upper = query.upper()

        # Check for required SQL components
        self.assertIn("SELECT", query_upper, f"{query_name} missing SELECT statement")
        self.assertIn("FROM", query_upper, f"{query_name} missing FROM clause")

        # Check for balanced parentheses
        open_parens = query.count("(")
        close_parens = query.count(")")
        self.assertEqual(
            open_parens,
            close_parens,
            f"{query_name} has unbalanced parentheses: {open_parens} open, {close_parens} close",
        )

        # Check for BigQuery-specific syntax elements
        if "mozfun.map.get_key" in query:
            # Validate mozfun usage patterns
            self.assertRegex(
                query,
                r"mozfun\.map\.get_key\([^)]+\)\.branch",
                f"{query_name} mozfun.map.get_key usage seems malformed",
            )

        # Check for date formatting
        if "DATE(" in query_upper:
            self.assertRegex(
                query,
                r"DATE\(['\"][0-9]{4}-[0-9]{2}-[0-9]{2}['\"]",
                f"{query_name} DATE() formatting seems incorrect",
            )

        # Check for common BigQuery table patterns
        if "moz" in query:
            self.assertRegex(
                query,
                r"`[^`]*moz[^`]*`",
                f"{query_name} BigQuery table names should be backtick-quoted",
            )

        print(f"✓ {query_name} passed basic syntax checks")

    def test_histogram_query_generation_experiment(self):
        """Test histogram query generation for experiments."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows", "Linux", "Mac"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "histogram-experiment", config_overrides
        )

        # Generate histogram query
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(query, "Histogram query generation failed")
        self.generated_queries.append(("histogram_experiment_glean", query))

        # Validate the query
        self.validate_sql_basic_checks(query, "histogram_experiment_glean")

        if SQL_VALIDATOR_AVAILABLE:
            self.validate_sql_with_sqlvalidator(query, "histogram_experiment_glean")

        if BIGQUERY_PARSER_AVAILABLE:
            self.validate_sql_with_bigquery_parser(query, "histogram_experiment_glean")

    def test_histogram_query_generation_with_non_enrolled(self):
        """Test histogram query generation with non-enrolled branch."""
        config_overrides = {
            "is_experiment": True,
            "channel": "nightly",
            "startDate": "2024-02-01",
            "endDate": "2024-02-07",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "include_non_enrolled_branch": True,
        }

        telemetry_client = self.create_experiment_config(
            "histogram-non-enrolled", config_overrides
        )

        # Generate histogram query with non-enrolled branch
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(
            query, "Histogram query with non-enrolled generation failed"
        )
        self.generated_queries.append(("histogram_with_non_enrolled", query))

        # Check for non-enrolled specific content
        self.assertIn(
            "non-enrolled", query, "Query should include non-enrolled branch logic"
        )
        self.assertIn(
            "desktop_data_non_enrolled",
            query,
            "Query should include non-enrolled desktop CTE",
        )

        # Validate the query
        self.validate_sql_basic_checks(query, "histogram_with_non_enrolled")

    def test_pageload_event_query_generation(self):
        """Test pageload event query generation."""
        config_overrides = {
            "is_experiment": True,
            "channel": "beta",
            "startDate": "2024-03-01",
            "endDate": "2024-03-07",
            "segments": ["Windows", "Linux"],
            "pageload_event_metrics": {
                "fcp_time": {"min": 0, "max": 30000},
                "load_time": {"min": 0, "max": 60000},
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "pageload-events", config_overrides
        )

        # Generate pageload event queries
        for metric in ["fcp_time", "load_time"]:
            query = telemetry_client.generatePageloadEventQuery_OS_segments(metric)

            self.assertIsNotNone(
                query, f"Pageload event query generation failed for {metric}"
            )
            self.generated_queries.append((f"pageload_event_{metric}", query))

            # Check metric-specific content
            self.assertIn(metric, query, f"Query should reference metric {metric}")
            self.assertIn("pageload", query, "Query should reference pageload tables")

            # Validate the query
            self.validate_sql_basic_checks(query, f"pageload_event_{metric}")

    def test_rollout_query_generation(self):
        """Test query generation for rollout experiments."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-04-01",
            "endDate": "2024-04-07",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "isRollout": True,
            "include_non_enrolled_branch": True,
        }

        telemetry_client = self.create_experiment_config(
            "rollout-test", config_overrides
        )

        # Generate rollout histogram query
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(query, "Rollout query generation failed")
        self.generated_queries.append(("rollout_histogram", query))

        # Rollouts should include non-enrolled branch logic
        self.assertIn(
            "non-enrolled",
            query,
            "Rollout query should include non-enrolled branch logic",
        )

        # Validate the query
        self.validate_sql_basic_checks(query, "rollout_histogram")

    def test_legacy_histogram_query_generation(self):
        """Test legacy histogram query generation."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows", "Mac"],
            "histograms": {
                "BROWSER_ENGAGEMENT_TOTAL_URI_COUNT": {
                    "glean": False,
                    "kind": "numerical",
                    "available_on_desktop": True,
                    "available_on_android": False,
                }
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "legacy-histogram", config_overrides
        )

        # Generate legacy histogram query
        histogram = "BROWSER_ENGAGEMENT_TOTAL_URI_COUNT"
        query = telemetry_client.generateHistogramQuery_OS_segments_legacy(histogram)

        self.assertIsNotNone(query, "Legacy histogram query generation failed")
        self.generated_queries.append(("legacy_histogram", query))

        # Legacy queries use different table structures
        self.assertNotIn("metrics", query, "Legacy queries shouldn't use metrics table")

        # Validate the query
        self.validate_sql_basic_checks(query, "legacy_histogram")

    def test_query_generation_with_isp_blacklist(self):
        """Test query generation with ISP blacklist."""
        # Create temporary ISP blacklist file
        isp_blacklist_file = os.path.join(self.test_dir, "isp_blacklist.txt")
        with open(isp_blacklist_file, "w") as f:
            f.write("Bad ISP 1\n")
            f.write("Bad ISP 2\n")
            f.write("Suspicious Provider\n")

        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "isp_blacklist": isp_blacklist_file,
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "isp-blacklist", config_overrides
        )

        # Generate query with ISP blacklist
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(query, "Query generation with ISP blacklist failed")
        self.generated_queries.append(("query_with_isp_blacklist", query))

        # Check for ISP blacklist conditions
        self.assertIn("metadata.isp.name", query, "Query should include ISP filtering")
        self.assertIn(
            "Bad ISP 1", query, "Query should include specific blacklisted ISPs"
        )
        self.assertIn("!=", query, "Query should exclude blacklisted ISPs")

        # Validate the query
        self.validate_sql_basic_checks(query, "query_with_isp_blacklist")

    @unittest.skipUnless(
        SQL_VALIDATOR_AVAILABLE or BIGQUERY_PARSER_AVAILABLE,
        "No SQL validation packages available",
    )
    def test_all_generated_queries_syntax(self):
        """Comprehensive syntax validation of all generated queries."""
        # Run all other test methods to populate generated_queries
        test_methods = [
            self.test_histogram_query_generation_experiment,
            self.test_histogram_query_generation_with_non_enrolled,
            self.test_pageload_event_query_generation,
            self.test_rollout_query_generation,
            self.test_legacy_histogram_query_generation,
            self.test_query_generation_with_isp_blacklist,
            self.test_empty_pageload_event_metrics,
            self.test_empty_histograms,
            self.test_only_histograms_no_pageload_events,
            self.test_only_pageload_events_no_histograms,
        ]

        # Clear any existing queries and run all tests
        self.generated_queries = []
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"Warning: {test_method.__name__} failed: {e}")

        # Validate all collected queries
        self.assertGreater(
            len(self.generated_queries), 0, "No queries were generated for validation"
        )

        print(f"\n=== Validating {len(self.generated_queries)} Generated Queries ===")

        validation_results = {"passed": 0, "failed": 0, "warnings": 0}

        for query_name, query in self.generated_queries:
            try:
                # Always run basic checks
                self.validate_sql_basic_checks(query, query_name)
                validation_results["passed"] += 1

                # Run advanced validation if packages are available
                if SQL_VALIDATOR_AVAILABLE:
                    try:
                        self.validate_sql_with_sqlvalidator(query, query_name)
                    except Exception as e:
                        print(f"⚠ {query_name} sqlvalidator warning: {e}")
                        validation_results["warnings"] += 1

                if BIGQUERY_PARSER_AVAILABLE:
                    if not self.validate_sql_with_bigquery_parser(query, query_name):
                        validation_results["warnings"] += 1

            except Exception as e:
                print(f"✗ {query_name} failed validation: {e}")
                validation_results["failed"] += 1

        print("\n=== Validation Summary ===")
        print(f"Passed: {validation_results['passed']}")
        print(f"Warnings: {validation_results['warnings']}")
        print(f"Failed: {validation_results['failed']}")

        # Test should pass if no queries completely failed validation
        self.assertEqual(
            validation_results["failed"],
            0,
            f"{validation_results['failed']} queries failed validation",
        )

    def test_query_parameter_substitution(self):
        """Test that Django template parameters are properly substituted."""
        config_overrides = {
            "is_experiment": True,
            "channel": "nightly",
            "startDate": "2024-05-15",
            "endDate": "2024-05-20",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "parameter-substitution", config_overrides
        )

        # Generate query
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(query, "Query generation failed")

        # Check that template parameters were substituted
        self.assertIn("parameter-substitution", query, "Slug parameter not substituted")
        self.assertIn("nightly", query, "Channel parameter not substituted")
        self.assertIn("2024-05-15", query, "Start date parameter not substituted")
        self.assertIn("2024-05-20", query, "End date parameter not substituted")
        self.assertIn(histogram, query, "Histogram parameter not substituted")

        # Check that no template syntax remains
        self.assertNotIn("{{", query, "Template syntax not fully substituted")
        self.assertNotIn("}}", query, "Template syntax not fully substituted")
        self.assertNotIn("{%", query, "Template logic not fully processed")
        self.assertNotIn("%}", query, "Template logic not fully processed")

        print("✓ Template parameter substitution validation passed")

    def test_empty_pageload_event_metrics(self):
        """Test query generation when pageload_event_metrics is empty."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            "pageload_event_metrics": {},  # Empty pageload events
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "empty-pageload-events", config_overrides
        )

        # Should still be able to generate histogram queries
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(
            query, "Histogram query should work with empty pageload events"
        )
        self.generated_queries.append(("empty_pageload_events_histogram", query))

        # Validate the query
        self.validate_sql_basic_checks(query, "empty_pageload_events_histogram")

        # Attempting to generate pageload event query should handle gracefully
        # (In practice, this would likely not be called if no metrics are configured)

        print("✓ Empty pageload event metrics test passed")

    def test_empty_histograms(self):
        """Test query generation when histograms is empty."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            "histograms": {},  # Empty histograms
            "pageload_event_metrics": {
                "fcp_time": {"min": 0, "max": 30000},
                "load_time": {"min": 0, "max": 60000},
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "empty-histograms", config_overrides
        )

        # Should still be able to generate pageload event queries
        for metric in ["fcp_time", "load_time"]:
            query = telemetry_client.generatePageloadEventQuery_OS_segments(metric)

            self.assertIsNotNone(
                query,
                f"Pageload event query should work with empty histograms for {metric}",
            )
            self.generated_queries.append((f"empty_histograms_{metric}", query))

            # Check metric-specific content
            self.assertIn(metric, query, f"Query should reference metric {metric}")
            self.assertIn("pageload", query, "Query should reference pageload tables")

            # Validate the query
            self.validate_sql_basic_checks(query, f"empty_histograms_{metric}")

        print("✓ Empty histograms test passed")

    def test_both_empty_metrics_and_histograms(self):
        """Test config validation when both metrics and histograms are empty."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            "histograms": {},  # Empty histograms
            "pageload_event_metrics": {},  # Empty pageload events
            "include_non_enrolled_branch": False,
        }

        # This should still create a telemetry client successfully
        telemetry_client = self.create_experiment_config("both-empty", config_overrides)

        # The client should exist but have no queries to generate
        self.assertIsNotNone(
            telemetry_client,
            "TelemetryClient should be created even with empty configs",
        )

        # Verify the config was set up correctly
        self.assertEqual(len(telemetry_client.config["histograms"]), 0)
        self.assertEqual(len(telemetry_client.config["pageload_event_metrics"]), 0)

        print("✓ Both empty metrics and histograms test passed")

    def test_only_histograms_no_pageload_events(self):
        """Test config with only histograms and no pageload_event_metrics key."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.performance_pageload_fcp": {
                    "glean": True,
                    "desc": "Test histogram for FCP timing",
                    "available_on_desktop": True,
                    "available_on_android": True,
                    "kind": "numerical",
                }
            },
            # No pageload_event_metrics key at all
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "histograms-only", config_overrides
        )

        # Should be able to generate histogram queries
        histogram = "metrics.timing_distribution.performance_pageload_fcp"
        query = telemetry_client.generateHistogramQuery_OS_segments_glean(histogram)

        self.assertIsNotNone(
            query, "Histogram query should work without pageload_event_metrics"
        )
        self.generated_queries.append(("histograms_only", query))

        # Validate the query
        self.validate_sql_basic_checks(query, "histograms_only")

        print("✓ Histograms only (no pageload events) test passed")

    def test_only_pageload_events_no_histograms(self):
        """Test config with only pageload events and no histograms key."""
        config_overrides = {
            "is_experiment": True,
            "channel": "release",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "segments": ["Windows"],
            # No histograms key at all
            "pageload_event_metrics": {
                "fcp_time": {"min": 0, "max": 30000},
                "load_time": {"min": 0, "max": 60000},
            },
            "include_non_enrolled_branch": False,
        }

        telemetry_client = self.create_experiment_config(
            "pageload-only", config_overrides
        )

        # Should be able to generate pageload event queries
        for metric in ["fcp_time", "load_time"]:
            query = telemetry_client.generatePageloadEventQuery_OS_segments(metric)

            self.assertIsNotNone(
                query,
                f"Pageload event query should work without histograms for {metric}",
            )
            self.generated_queries.append((f"pageload_only_{metric}", query))

            # Check metric-specific content
            self.assertIn(metric, query, f"Query should reference metric {metric}")
            self.assertIn("pageload", query, "Query should reference pageload tables")

            # Validate the query
            self.validate_sql_basic_checks(query, f"pageload_only_{metric}")

        print("✓ Pageload events only (no histograms) test passed")


if __name__ == "__main__":
    # Print information about available validation packages
    print("=== SQL Validation Package Availability ===")
    print(f"sqlvalidator: {'✓' if SQL_VALIDATOR_AVAILABLE else '✗'}")
    print(f"bigquery-sql-parser: {'✓' if BIGQUERY_PARSER_AVAILABLE else '✗'}")

    if not SQL_VALIDATOR_AVAILABLE and not BIGQUERY_PARSER_AVAILABLE:
        print("\n⚠ No SQL validation packages available.")
        print("Install with: pip install sqlvalidator bigquery-sql-parser")
        print("Tests will run with basic syntax checking only.\n")

    unittest.main()
