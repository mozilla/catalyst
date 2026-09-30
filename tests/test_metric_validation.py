"""Regression coverage for invalid histogram weights (bug 2076833)."""

from decimal import Decimal
import warnings

import numpy as np
import pytest

from lib.analysis import DataAnalyzer
from lib.generate import isValidMetricData, transformTelemetryDataByType


@pytest.mark.parametrize(
    "bins,counts",
    [
        ([0, 100], [200002, -2]),
        ([0, 100], [Decimal("200002"), Decimal("-2")]),
        ([0, 100], [-1, -2]),
        ([0, 100], [1, float("nan")]),
        ([0, 100], [1, float("inf")]),
        ([0, float("inf")], [1, 2]),
        ([0, 100], [1]),
        ([0, 100], [0, 0]),
        ([], []),
    ],
)
def test_invalid_histogram_is_skipped(bins, counts, capsys):
    assert not isValidMetricData(
        {"bins": bins, "counts": counts},
        "latency",
        "control",
        "Windows",
        "histograms",
        "numerical",
    )
    assert "Skipping control/Windows: histograms.latency" in capsys.readouterr().out


def test_valid_decimal_counts_and_zero_buckets():
    assert isValidMetricData(
        {"bins": [0, 100], "counts": [Decimal("0"), Decimal("200002")]},
        "latency",
        "control",
        "Windows",
        "histograms",
        "numerical",
    )
    assert isValidMetricData(
        {"bins": ["parent", "content"], "counts": [0, 10]},
        "process",
        "control",
        "Windows",
        "histograms",
        "categorical",
    )


@pytest.mark.parametrize("invalid_branch", ["control", "treatment"])
def test_negative_counts_do_not_reach_analysis(invalid_branch, capsys):
    config = {
        "branches": ["control", "treatment"],
        "segments": ["Windows"],
        "histograms": {
            "latency": {"kind": "numerical"},
            "healthy": {"kind": "numerical"},
        },
    }
    telemetry = {}
    for branch in config["branches"]:
        telemetry[branch] = {
            "Windows": {
                "histograms": {
                    "latency": {
                        "bins": [0, 100],
                        "counts": (
                            [200002, -2] if branch == invalid_branch else [100, 200]
                        ),
                    },
                    "healthy": {"bins": [10, 20], "counts": [100, 200]},
                }
            }
        }

    # This input previously produced a negative variance, then crashed np.repeat.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        transformed = transformTelemetryDataByType(telemetry, config)
        results = DataAnalyzer(config).processTelemetryData(transformed)

    assert "latency" not in results[invalid_branch]["Windows"]["numerical"]
    valid_branch = "treatment" if invalid_branch == "control" else "control"
    latency = results[valid_branch]["Windows"]["numerical"]["latency"]
    assert latency["tests"] == {}  # No comparison against an invalid branch.
    assert np.isfinite(latency["std"])
    healthy = results["treatment"]["Windows"]["numerical"]["healthy"]
    assert healthy["n"] == 300
    assert "mwu" in healthy["tests"]
    assert (
        f"Skipping {invalid_branch}/Windows: histograms.latency"
        in capsys.readouterr().out
    )
