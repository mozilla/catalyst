#!/usr/bin/env python3
"""
Comprehensive config validation test suite.

This test validates all config files through the complete parsing and annotation
pipeline to catch field name errors, schema violations, and other issues.
"""

import sys
import os
import unittest
import glob
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.parser import parseConfigFile, annotateMetrics

# List of config files to skip during validation
SKIP_CONFIGS = [
    "configs/impact-of-race-cache-with-network-beta_has_ssd_false.yaml",
]


class TestConfigValidation(unittest.TestCase):
    """Test config file validation and schema compliance."""

    def setUp(self):
        """Set up test environment."""
        # Get all config files
        all_config_files = glob.glob("configs/*.yaml")

        # Filter out skipped configs
        self.config_files = [
            config for config in all_config_files if config not in SKIP_CONFIGS
        ]

        # Store skipped configs for reporting
        self.skipped_configs = [
            config for config in all_config_files if config in SKIP_CONFIGS
        ]

        self.assertTrue(
            len(self.config_files) > 0, "No config files found after filtering"
        )

    def validate_config_structure(self, config: Dict[str, Any], config_file: str):
        """Validate that a config has required fields and proper structure."""
        # Required fields
        required_fields = ["slug"]
        for field in required_fields:
            self.assertIn(
                field, config, f"{config_file}: Missing required field '{field}'"
            )

        # Must have either histograms or pageload_event_metrics (or both)
        has_histograms = "histograms" in config and config["histograms"]
        has_pageload_metrics = (
            "pageload_event_metrics" in config and config["pageload_event_metrics"]
        )

        self.assertTrue(
            has_histograms or has_pageload_metrics,
            f"{config_file}: Must have either histograms or pageload_event_metrics (or both)",
        )

        # Validate histograms field format
        if "histograms" in config:
            histograms = config["histograms"]
            self.assertIsInstance(
                histograms, list, f"{config_file}: histograms must be a list"
            )

        # Validate pageload_event_metrics field format
        if "pageload_event_metrics" in config:
            metrics = config["pageload_event_metrics"]
            self.assertIsInstance(
                metrics, dict, f"{config_file}: pageload_event_metrics must be a dict"
            )

            for metric_name, metric_config in metrics.items():
                self.assertIsInstance(
                    metric_config,
                    dict,
                    f"{config_file}: {metric_name} must be a dict with max parameter",
                )
                self.assertIn(
                    "max",
                    metric_config,
                    f"{config_file}: {metric_name} must have 'max' parameter",
                )

        # Validate segments field
        if "segments" in config:
            segments = config["segments"]
            # Segments can be a list or a dict (for non-experiment configs)
            if isinstance(segments, list):
                # For list format, validate segment names
                allowed_segments = ["Windows", "Linux", "Mac", "Android", "All"]
                for segment in segments:
                    if segment not in allowed_segments:
                        print(
                            f"Warning: {config_file} uses unrecognized segment '{segment}'"
                        )
            elif isinstance(segments, dict):
                # Dict format is valid for non-experiment configs
                pass
            else:
                self.fail(
                    f"{config_file}: segments must be a list or dict, got {type(segments)}"
                )

        # Check for common typos in field names
        invalid_fields = []
        for field in config.keys():
            if (
                field.startswith("pageload_event_metrics")
                and field != "pageload_event_metrics"
            ):
                invalid_fields.append(field)
            elif field.startswith("histogram") and field not in ["histograms"]:
                invalid_fields.append(field)

        if invalid_fields:
            self.fail(
                f"{config_file}: Invalid field names (possible typos): {invalid_fields}"
            )

    def test_all_config_files_parse_and_validate(self):
        """Test that all config files parse correctly and pass field name validation."""
        success_count = 0
        failed_configs = []

        print("")
        for config_file in self.config_files:
            try:
                # Step 1: Parse the config file
                config = parseConfigFile(config_file)

                # Step 2: Validate config structure
                self.validate_config_structure(config, config_file)

                # Step 3: Check for field name typos (main focus)
                self._check_field_name_typos(config, config_file)

                # Step 4: Run annotation if config has metrics (this catches field name errors)
                if "pageload_event_metrics" in config or "histograms" in config:
                    annotateMetrics(config)

                print(f"✅ {config_file}")
                success_count += 1

            except Exception as e:
                print(f"❌ {config_file}: {e}")
                failed_configs.append((config_file, str(e)))

        print(
            f"\nValidation Summary: {success_count}/{len(self.config_files)} configs passed"
        )

        if self.skipped_configs:
            print(f"Skipped: {len(self.skipped_configs)} configs")
            for config_file in self.skipped_configs:
                print(f"  - {config_file} (intentionally skipped)")

        if failed_configs:
            print("\nFailed configurations:")
            for config_file, error in failed_configs:
                print(f"  - {config_file}: {error}")

        # Test should fail if any configs have field name errors
        self.assertEqual(
            len(failed_configs),
            0,
            f"{len(failed_configs)} config files failed validation: {[c[0] for c in failed_configs]}",
        )

    def _check_field_name_typos(self, config: Dict[str, Any], config_file: str):
        """Check for common typos in field names."""
        invalid_fields = []
        for field in config.keys():
            if (
                field.startswith("pageload_event_metrics")
                and field != "pageload_event_metrics"
            ):
                invalid_fields.append(field)
            elif field.startswith("histogram") and field not in ["histograms"]:
                invalid_fields.append(field)

        if invalid_fields:
            raise ValueError(f"Invalid field names (possible typos): {invalid_fields}")

    def test_field_name_validation_only(self):
        """Test focused specifically on field name validation."""
        success_count = 0
        field_name_errors = []

        for config_file in self.config_files:
            try:
                config = parseConfigFile(config_file)
                self._check_field_name_typos(config, config_file)
                success_count += 1

            except ValueError as e:
                print(f"❌ {config_file}: {e}")
                field_name_errors.append((config_file, str(e)))
            except Exception:
                # Skip other parsing errors, focus only on field names
                success_count += 1

        print(
            f"\nField Name Validation: {success_count}/{len(self.config_files)} configs have correct field names"
        )

        if self.skipped_configs:
            print(f"Skipped: {len(self.skipped_configs)} configs")
            for config_file in self.skipped_configs:
                print(f"  - {config_file} (intentionally skipped)")

        if field_name_errors:
            print("\nConfigs with field name errors:")
            for config_file, error in field_name_errors:
                print(f"  - {config_file}: {error}")

        # This test specifically focuses on field name typos
        self.assertEqual(
            len(field_name_errors),
            0,
            f"{len(field_name_errors)} config files have field name errors: {[c[0] for c in field_name_errors]}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
