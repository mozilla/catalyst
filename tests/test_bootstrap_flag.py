"""enable_bootstrap_ci gates bootstrap CIs, the sample-size floor, and summary colors."""

import numpy as np
import pandas as pd
import pytest

from lib.analysis import DataAnalyzer
from lib.generate import resultsMatchBootstrapFlag, transformTelemetryDataByType
from lib.report import get_ci_significance, get_summary_color
from lib.telemetry import TelemetryClient


@pytest.mark.parametrize("with_replicates", [False, True])
def test_lift_ci_requires_replicates(with_replicates):
    config = {
        "slug": "flag-test",
        "branches": ["control", "treatment"],
        "segments": ["Windows"],
        "histograms": {"latency": {"kind": "numerical"}},
    }
    counts = [4000, 3000, 2000, 1000]
    histogram = {"bins": [10, 20, 40, 80], "counts": counts}
    if with_replicates:
        rng = np.random.default_rng(0)
        histogram["bootstrap_replicates"] = [
            c * rng.exponential(size=50) for c in counts
        ]
    telemetry = {
        branch: {"Windows": {"histograms": {"latency": dict(histogram)}}}
        for branch in config["branches"]
    }

    transformed = transformTelemetryDataByType(telemetry, config)
    results = DataAnalyzer(config).processTelemetryData(transformed)

    latency = results["treatment"]["Windows"]["numerical"]["latency"]
    assert "mwu" in latency["tests"]  # significance tests run either way
    for key in ("median_lift_ci", "p75_lift_ci", "p95_lift_ci"):
        assert (latency[key]["lower"] is not None) == with_replicates


@pytest.mark.parametrize("enabled", [False, True])
def test_bootstrap_fetch_follows_flag(enabled, tmp_path):
    name = "metrics.timing_distribution.latency"
    config = {
        "slug": "s",
        "segments": ["Windows"],
        "histograms": {
            name: {
                "kind": "numerical",
                "glean": True,
                "available_on_desktop": True,
                "available_on_android": False,
            }
        },
        "enable_bootstrap_ci": enabled,
    }
    client = TelemetryClient(str(tmp_path), config)
    fetched = []
    client.getHistogramDataClientArrays = lambda config, hists: fetched.extend(hists)
    client._prefetchBatchedNumericalHistograms(config["histograms"])
    assert fetched == ([name] if enabled else [])


class _Job:
    """Stands in for a BigQuery job; flat consumers must never get Arrow."""

    def to_dataframe(self):
        return pd.DataFrame({"x": [1]})

    def result(self):
        pytest.fail("flat fetch must not stream Arrow")


@pytest.mark.parametrize(
    "fetch, generator",
    [
        (
            "getPageloadEventDataNonExperiment",
            "generatePageloadEventQuery_OS_segments_non_experiment",
        ),
        (
            "getCrashEventDataNonExperiment",
            "generateCrashEventQuery_OS_segments_non_experiment",
        ),
        ("getCrashEventData", "generateCrashEventQuery_OS_segments"),
    ],
)
def test_flat_fetches_stay_dataframes_flag_on(fetch, generator, tmp_path):
    """Only the bootstrap fetches read Arrow; crash events and non-experiment
    pageload events feed pandas consumers even with the flag on."""
    config = {
        "slug": "s",
        "segments": {"Windows": {}, "Mac": {}, "Linux": {}},
        "enable_bootstrap_ci": True,
    }
    client = TelemetryClient(str(tmp_path), config)
    client._client = type("C", (), {"query": lambda self, q: _Job()})()
    setattr(client, generator, lambda metric: "SELECT 1")
    assert isinstance(getattr(client, fetch)("m"), pd.DataFrame)


@pytest.mark.parametrize("enabled", [False, True])
def test_pageload_fetch_reads_arrow_only_with_flag_on(enabled, tmp_path, monkeypatch):
    """The pageload fetch is the one that feeds the bootstrap engine, which
    takes Arrow; flag off feeds pandas consumers."""
    import lib.telemetry as telemetry

    read = []

    class Job:
        def to_dataframe(self):
            read.append("dataframe")
            return pd.DataFrame({"x": [1]})

    config = {
        "slug": "s",
        "segments": {"Windows": {}, "Mac": {}, "Linux": {}},
        "enable_bootstrap_ci": enabled,
    }
    client = TelemetryClient(str(tmp_path), config)
    client._client = type("C", (), {"query": lambda self, q: Job()})()
    client.generatePageloadEventQuery_OS_segments = lambda metric: "SELECT 1"
    client.generatePageloadEventQuery_OS_segments_document_arrays = (
        lambda metric: "SELECT 1"
    )
    monkeypatch.setattr(telemetry, "arrowBatches", lambda job: read.append("arrow"))
    monkeypatch.setattr(
        telemetry, "compute_client_cluster_bootstrap", lambda df, **kw: pd.DataFrame()
    )
    client.getPageloadEventData("m")
    assert read == (["arrow"] if enabled else ["dataframe"])


def test_skip_cache_reuses_batched_fetch_in_memory(tmp_path):
    """--skip-cache disables cache reads, so the per-metric read after the
    batched prefetch must come from memory, not a second query."""
    name = "metrics.timing_distribution.latency"
    config = {
        "slug": "s",
        "segments": ["Windows"],
        "histograms": {name: {"kind": "numerical", "glean": True}},
        "enable_bootstrap_ci": True,
    }
    client = TelemetryClient(str(tmp_path), config, skipCache=True)
    df = pd.DataFrame({"bucket": [1.0]})
    client.prefetchedHistograms[name] = df
    client.generateHistogramQuery_OS_segments_glean = lambda h: pytest.fail("refetched")
    assert client.getHistogramData(config, name) is df


@pytest.mark.parametrize("enabled", [False, True])
def test_sample_size_floor_follows_flag(enabled, tmp_path):
    config = {
        "branches": [{"name": "control"}, {"name": "treatment"}],
        "segments": ["Windows"],
        "histograms": {
            "latency": {"distribution_type": "timing_distribution", "kind": "numerical"}
        },
        "enable_bootstrap_ci": enabled,
    }
    df = pd.DataFrame(
        {
            "segment": ["Windows"] * 2,
            "branch": ["control", "treatment"],
            "counts": [10, 10],
        }
    )
    client = TelemetryClient(str(tmp_path), config)
    assert bool(client._insufficientSampleSegments(df, "latency")) == enabled


def test_cache_files_differ_by_flag(tmp_path):
    def client(enabled):
        config = {
            "slug": "s",
            "segments": ["Windows"],
            "histograms": {
                "metrics.timing_distribution.latency": {
                    "kind": "numerical",
                    "glean": True,
                },
                "metrics.labeled_counter.process": {
                    "kind": "categorical",
                    "glean": True,
                },
            },
            "enable_bootstrap_ci": enabled,
        }
        return TelemetryClient(str(tmp_path), config)

    off, on = client(False), client(True)
    numerical = "metrics.timing_distribution.latency"
    categorical = "metrics.labeled_counter.process"
    assert off._histogramCacheFile(numerical) != on._histogramCacheFile(numerical)
    assert off._histogramCacheFile(categorical) == on._histogramCacheFile(categorical)
    assert off._pageloadEventCacheFile("lcp_time") != on._pageloadEventCacheFile(
        "lcp_time"
    )


def test_saved_results_require_matching_flag():
    saved = {"input": {"enable_bootstrap_ci": True}}
    assert resultsMatchBootstrapFlag(saved, {"enable_bootstrap_ci": True})
    assert not resultsMatchBootstrapFlag(saved, {"enable_bootstrap_ci": False})
    assert not resultsMatchBootstrapFlag(saved, {})
    assert resultsMatchBootstrapFlag({"input": {}}, {})


NORMAL = "font-weight: normal"
GREEN = "font-weight: bold; color: green"
RED = "font-weight: bold; color: red"


@pytest.mark.parametrize(
    "uplift_pct,lift_ci,higher_is_better,expected",
    [
        # CI excludes zero and lift clears 1.5%: colored by direction.
        (-3.0, {"lower": -0.05, "upper": -0.01}, False, GREEN),
        (3.0, {"lower": 0.01, "upper": 0.05}, False, RED),
        (3.0, {"lower": 0.01, "upper": 0.05}, True, GREEN),
        # CI spans zero: not significant, even with a large lift.
        (5.0, {"lower": -0.01, "upper": 0.11}, False, NORMAL),
        # Significant but below the 1.5% practical threshold.
        (1.0, {"lower": 0.002, "upper": 0.018}, False, NORMAL),
        # No CI available: nothing to base significance on.
        (5.0, None, False, NORMAL),
        (5.0, {"lower": None, "upper": None}, False, NORMAL),
    ],
)
def test_summary_color_uses_lift_ci_when_flag_on(
    uplift_pct, lift_ci, higher_is_better, expected
):
    # effect_meaning is the MWU verdict; flag on, it must not affect color.
    for effect_meaning in ("None", "Large"):
        assert (
            get_summary_color(
                uplift_pct, lift_ci, effect_meaning, higher_is_better, True
            )
            == expected
        )


@pytest.mark.parametrize(
    "uplift_pct,effect_meaning,expected",
    [
        (3.0, "Large", RED),
        (-3.0, "Medium", GREEN),
        (3.0, "Small", NORMAL),
        (3.0, "None", NORMAL),
        (1.0, "Large", NORMAL),
    ],
)
def test_summary_color_uses_mwu_when_flag_off(uplift_pct, effect_meaning, expected):
    # Flag off ignores the lift CI, matching main.
    ci = {"lower": -0.01, "upper": 0.05}
    assert get_summary_color(uplift_pct, ci, effect_meaning, False, False) == expected


def test_ci_significance_lists_percentiles_excluding_zero():
    excludes = {"lower": 0.01, "upper": 0.05}
    spans = {"lower": -0.01, "upper": 0.05}
    assert (
        get_ci_significance(
            {"median_lift_ci": excludes, "p75_lift_ci": spans, "p95_lift_ci": excludes}
        )
        == "median, p95"
    )
    assert get_ci_significance({"median_lift_ci": spans}) == "None"
    assert get_ci_significance({"median_lift_ci": {"lower": None}}) == "No CI"
    assert get_ci_significance({}) == "No CI"
