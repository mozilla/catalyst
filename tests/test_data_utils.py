#!/usr/bin/env python3
"""
Shared utilities for creating artificial test data for Catalyst experiments.

This module provides functions to generate realistic artificial telemetry data
for testing purposes, including histogram data and pageload event data.
"""

import os
import json
import pandas as pd
import numpy as np


def create_histogram_data(
    data_dir,
    slug,
    full_metric_name,
    branches=None,
    segments=None,
    branch_stats=None,
    total_samples=1000,
):
    """Create artificial histogram data with realistic distributions.

    Args:
        data_dir: Directory to save the data file
        slug: Experiment slug for filename
        full_metric_name: Full metric name (e.g., "metrics.timing_distribution.performance_pageload_fcp")
        branches: List of branch names (defaults to ["control", "treatment"])
        segments: List of segment names (defaults to ["Windows", "Linux", "Mac"])
        branch_stats: Dictionary mapping branch names to {"mean": float, "median": float, "stddev": float}
                     If None, uses default patterns (control slower, treatment faster)
        total_samples: Total number of samples to generate per branch/segment combination
    """
    if branches is None:
        branches = ["control", "treatment"]
    if segments is None:
        segments = ["Windows", "Linux", "Mac"]

    # Set default statistical parameters if not provided
    if branch_stats is None:
        branch_stats = {
            "control": {"mean": 2000, "median": 1500, "stddev": 1200},
            "treatment": {"mean": 1200, "median": 1000, "stddev": 800},
            "default": {"mean": 1500, "median": 1200, "stddev": 1000},
        }

    # Generate data for each branch and segment
    data_rows = []

    # Extract the last part of the metric name for filename
    metric_name = full_metric_name.split(".")[-1]

    for branch in branches:
        for segment in segments:
            # Get branch statistics or use default pattern
            if branch in branch_stats:
                stats_config = branch_stats[branch]
            elif branch == "control":
                stats_config = {"mean": 2000, "median": 1500, "stddev": 1200}
            elif branch == "treatment":
                stats_config = {"mean": 1200, "median": 1000, "stddev": 800}
            else:
                # Default for unknown branches
                stats_config = {"mean": 1500, "median": 1200, "stddev": 1000}

            mean = stats_config["mean"]
            stddev = stats_config["stddev"]

            # Add segment-specific variation to the statistics
            segment_multiplier = 1.0
            if segment == "Windows":
                segment_multiplier = 1.1  # Slightly slower on Windows
            elif segment == "Mac":
                segment_multiplier = 0.9  # Slightly faster on Mac

            adjusted_mean = mean * segment_multiplier
            adjusted_stddev = stddev * segment_multiplier

            # Generate a log-normal distribution to match typical performance data
            # Log-normal is good for timing data as it's always positive and right-skewed

            # Calculate log-normal parameters from desired mean and stddev
            # For log-normal: mean = exp(mu + sigma^2/2), var = exp(2*mu + sigma^2) * (exp(sigma^2) - 1)
            variance = adjusted_stddev**2
            mu = np.log(adjusted_mean**2 / np.sqrt(variance + adjusted_mean**2))
            sigma = np.sqrt(np.log(1 + variance / adjusted_mean**2))

            # Generate samples from log-normal distribution
            samples = np.random.lognormal(mu, sigma, total_samples)

            # Ensure samples are within reasonable bounds (positive and not too extreme)
            samples = np.clip(samples, 10, 50000)

            # Create histogram buckets based on the data range
            min_val = max(10, int(np.min(samples) * 0.8))
            max_val = min(50000, int(np.max(samples) * 1.2))

            # Create exponentially spaced buckets for better distribution
            bucket_edges = np.logspace(np.log10(min_val), np.log10(max_val), 20)
            bucket_edges = np.unique(np.round(bucket_edges).astype(int))

            # Create histogram from samples
            hist_counts, bin_edges = np.histogram(samples, bins=bucket_edges)

            # Use bucket centers as the bucket values
            buckets = [
                (bin_edges[i] + bin_edges[i + 1]) // 2 for i in range(len(hist_counts))
            ]

            for bucket, count in zip(buckets, hist_counts):
                if count > 0:  # Only add non-zero counts
                    data_rows.append(
                        {
                            "segment": segment,
                            "branch": branch,
                            "bucket": int(bucket),
                            "counts": int(count),
                        }
                    )

    # Save as pickle file
    df = pd.DataFrame(data_rows)
    filename = os.path.join(data_dir, f"{slug}-{metric_name}.pkl")
    df.to_pickle(filename)


def create_pageload_event_data(
    data_dir,
    slug,
    metric_name,
    branches=None,
    segments=None,
    branch_stats=None,
    total_samples=800,
):
    """Create artificial pageload event data with realistic distributions.

    Args:
        data_dir: Directory to save the data file
        slug: Experiment slug for filename
        metric_name: Event metric name (e.g., "fcp_time", "load_time")
        branches: List of branch names (defaults to ["control", "treatment"])
        segments: List of segment names (defaults to ["Windows", "Linux", "Mac"])
        branch_stats: Dictionary mapping branch names to {"mean": float, "median": float, "stddev": float}
                     If None, uses defaults based on metric_name
        total_samples: Total number of samples to generate per branch/segment combination
    """
    if branches is None:
        branches = ["control", "treatment"]
    if segments is None:
        segments = ["Windows", "Linux", "Mac"]

    # Set default statistical parameters based on metric_name if not provided
    if branch_stats is None:
        if metric_name == "fcp_time":
            branch_stats = {
                "control": {"mean": 1800, "median": 1500, "stddev": 800},
                "treatment": {"mean": 1400, "median": 1200, "stddev": 600},
            }
        elif metric_name == "load_time":
            branch_stats = {
                "control": {"mean": 4000, "median": 3500, "stddev": 1500},
                "treatment": {"mean": 3200, "median": 2800, "stddev": 1200},
            }
        else:
            # Generic event metrics
            branch_stats = {
                "control": {"mean": 2000, "median": 1700, "stddev": 900},
                "treatment": {"mean": 1500, "median": 1300, "stddev": 700},
            }

    # Generate data for each branch and segment
    data_rows = []

    for branch in branches:
        for segment in segments:
            # Get branch statistics or use default pattern
            if branch in branch_stats:
                stats_config = branch_stats[branch]
            elif branch == "control":
                if metric_name == "fcp_time":
                    stats_config = {"mean": 1800, "median": 1500, "stddev": 800}
                elif metric_name == "load_time":
                    stats_config = {"mean": 4000, "median": 3500, "stddev": 1500}
                else:
                    stats_config = {"mean": 2000, "median": 1700, "stddev": 900}
            elif branch == "treatment":
                if metric_name == "fcp_time":
                    stats_config = {"mean": 1400, "median": 1200, "stddev": 600}
                elif metric_name == "load_time":
                    stats_config = {"mean": 3200, "median": 2800, "stddev": 1200}
                else:
                    stats_config = {"mean": 1500, "median": 1300, "stddev": 700}
            else:
                # Default for unknown branches
                if metric_name == "fcp_time":
                    stats_config = {"mean": 1600, "median": 1350, "stddev": 700}
                elif metric_name == "load_time":
                    stats_config = {"mean": 3600, "median": 3150, "stddev": 1350}
                else:
                    stats_config = {"mean": 1750, "median": 1500, "stddev": 800}

            mean = stats_config["mean"]
            stddev = stats_config["stddev"]

            # Add segment-specific variation to the statistics
            segment_multiplier = 1.0
            if segment == "Windows":
                segment_multiplier = 1.1  # Slightly slower on Windows
            elif segment == "Mac":
                segment_multiplier = 0.9  # Slightly faster on Mac

            adjusted_mean = mean * segment_multiplier
            adjusted_stddev = stddev * segment_multiplier

            # Generate a log-normal distribution to match typical performance data
            # Log-normal is good for timing data as it's always positive and right-skewed

            # Calculate log-normal parameters from desired mean and stddev
            # For log-normal: mean = exp(mu + sigma^2/2), var = exp(2*mu + sigma^2) * (exp(sigma^2) - 1)
            variance = adjusted_stddev**2
            mu = np.log(adjusted_mean**2 / np.sqrt(variance + adjusted_mean**2))
            sigma = np.sqrt(np.log(1 + variance / adjusted_mean**2))

            # Generate samples from log-normal distribution
            samples = np.random.lognormal(mu, sigma, total_samples)

            # Ensure samples are within reasonable bounds for event metrics
            min_bound = 50  # 50ms minimum
            max_bound = 30000  # 30s maximum
            samples = np.clip(samples, min_bound, max_bound)

            # Create histogram buckets based on the data range
            min_val = max(min_bound, int(np.min(samples) * 0.8))
            max_val = min(max_bound, int(np.max(samples) * 1.2))

            # Create exponentially spaced buckets for better distribution
            bucket_edges = np.logspace(np.log10(min_val), np.log10(max_val), 20)
            bucket_edges = np.unique(np.round(bucket_edges).astype(int))

            # Create histogram from samples
            hist_counts, bin_edges = np.histogram(samples, bins=bucket_edges)

            # Use bucket centers as the bucket values
            buckets = [
                (bin_edges[i] + bin_edges[i + 1]) // 2 for i in range(len(hist_counts))
            ]

            for bucket, count in zip(buckets, hist_counts):
                if count > 0:  # Only add non-zero counts
                    data_rows.append(
                        {
                            "segment": segment,
                            "branch": branch,
                            "bucket": int(bucket),
                            "counts": int(count),
                        }
                    )

    # Save as pickle file
    df = pd.DataFrame(data_rows)
    filename = os.path.join(data_dir, f"{slug}-pageload-events-{metric_name}.pkl")
    df.to_pickle(filename)


def create_nimbus_api_cache(data_dir, slug, experiment_config):
    """Create cached Nimbus API response for an experiment.

    Args:
        data_dir: Directory to save the API cache file
        slug: Experiment slug
        experiment_config: Dictionary with experiment configuration including:
            - name: Experiment name
            - description: Experiment description
            - branches: List of branch dictionaries
            - startDate: Start date string
            - endDate: End date string
            - channels: List of channels
            - isRollout: Boolean indicating if this is a rollout
            - status: Experiment status (e.g., "Complete")
    """
    api_response = {
        "slug": slug,
        "name": experiment_config.get("name", f"Test Experiment {slug}"),
        "description": experiment_config.get(
            "description", f"A test experiment for {slug}"
        ),
        "branches": experiment_config.get(
            "branches",
            [
                {
                    "name": "control",
                    "slug": "control",
                    "description": "Control branch",
                    "ratio": 50,
                },
                {
                    "name": "treatment",
                    "slug": "treatment",
                    "description": "Treatment branch",
                    "ratio": 50,
                },
            ],
        ),
        "startDate": experiment_config.get("startDate", "2024-01-01"),
        "endDate": experiment_config.get("endDate", "2024-01-07"),
        "channels": experiment_config.get("channels", ["release"]),
        "isRollout": experiment_config.get("isRollout", False),
        "status": experiment_config.get("status", "Complete"),
    }

    api_cache_file = os.path.join(data_dir, f"{slug}-nimbus-API.json")
    with open(api_cache_file, "w") as f:
        json.dump(api_response, f, indent=2)


def create_test_config(slug, config_overrides=None):
    """Create a test experiment configuration.

    Args:
        slug: Experiment slug
        config_overrides: Dictionary of config values to override defaults

    Returns:
        Dictionary with experiment configuration
    """
    default_config = {
        "slug": slug,
        "channel": "release",
        "startDate": "2024-01-01",
        "endDate": "2024-01-07",
        "segments": ["Windows", "Linux", "Mac"],
        "is_experiment": True,
        "histograms": ["metrics.timing_distribution.performance_pageload_fcp"],
        "events": [
            {
                "pageload": {
                    "fcp_time": {"max": 30000},
                    "load_time": {"max": 30000},
                }
            }
        ],
    }

    if config_overrides:
        default_config.update(config_overrides)

    return default_config
