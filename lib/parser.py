import requests
import json
import yaml
import sys
import os
import datetime
from typing import Dict, Any, Optional


def checkForLocalFile(filename: str) -> Optional[Dict[str, Any]]:
    """Check if a local file exists and load its JSON content.

    Args:
      filename: Path to the file to check

    Returns:
      Dictionary containing the file's JSON data, or None if file cannot be loaded
    """
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            return data
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return None


def loadProbeIndex() -> Optional[Dict[str, Any]]:
    """Load the probe index from the local JSON file.

    Returns:
      Dictionary containing probe index data, or None if file cannot be loaded
    """
    filename = os.path.join(os.path.dirname(__file__), "probe-index.json")
    data = checkForLocalFile(filename)
    return data


def annotateMetrics(config: Dict[str, Any]) -> None:
    """Annotate config with probe index metadata for histograms and events.

    Args:
      config: Configuration dictionary to annotate
    """
    probeIndex = loadProbeIndex()
    annotateHistograms(config, probeIndex)

    # Handle new events structure
    if "events" in config:
        parseEventsConfiguration(config, probeIndex)
    else:
        # No events specified - create empty structures
        config["pageload_event_metrics"] = {}
        config["crash_event_metrics"] = {}


def annotatePageloadEventMetrics(
    config: Dict[str, Any], probeIndex: Dict[str, Any]
) -> None:
    """Annotate pageload event metrics with schema information.

    Args:
      config: Configuration dictionary containing pageload_event_metrics
      probeIndex: Probe index dictionary containing schema information
    """
    event_schema = probeIndex["glean"]["perf_page_load"]["extra_keys"]

    event_metrics = config["pageload_event_metrics"].copy()
    config["pageload_event_metrics"] = {}

    for metric in event_metrics:
        config["pageload_event_metrics"][metric] = {}
        if metric in event_schema:
            config["pageload_event_metrics"][metric]["desc"] = event_schema[metric][
                "description"
            ]
            config["pageload_event_metrics"][metric]["min"] = 0
            config["pageload_event_metrics"][metric][
                "kind"
            ] = "numerical"  # Pageload events are numerical histograms

            # Expect new max parameter format: {max: 30000}
            metric_config = event_metrics[metric]
            if isinstance(metric_config, dict):
                config["pageload_event_metrics"][metric]["max"] = metric_config.get(
                    "max", 30000
                )
            else:
                print(
                    f"ERROR: {metric} must use max parameter format, "
                    f"got: {metric_config}"
                )
                sys.exit(1)
        else:
            print(f"ERROR: {metric} not found in pageload event schema.")
            sys.exit(1)


def annotateCrashEventMetrics(
    config: Dict[str, Any], probeIndex: Dict[str, Any]
) -> None:
    """Annotate crash event metrics with basic configuration.

    Args:
      config: Configuration dictionary containing crash_event_metrics
      probeIndex: Probe index dictionary (currently unused for crash events)
    """
    if "crash_event_metrics" not in config:
        config["crash_event_metrics"] = {}
        return

    crash_metrics = config["crash_event_metrics"].copy()
    config["crash_event_metrics"] = {}

    for metric in crash_metrics:
        config["crash_event_metrics"][metric] = {
            "desc": "Total count of crashes for this experiment branch",
            "kind": "scalar",  # Crash events are scalar counts
        }


def parseEventsConfiguration(
    config: Dict[str, Any], probeIndex: Dict[str, Any]
) -> None:
    """Parse the new events configuration structure and populate legacy format.

    Args:
        config: Configuration dictionary containing events
        probeIndex: Probe index dictionary containing schema information
    """
    # Initialize empty legacy structures
    config["pageload_event_metrics"] = {}
    config["crash_event_metrics"] = {}

    if "events" not in config or not config["events"]:
        return

    # Default pageload metrics if pageload is specified without sub-metrics
    default_pageload_metrics = {
        "fcp_time": {"max": 30000},
        "lcp_time": {"max": 30000},
        "load_time": {"max": 30000},
        "response_time": {"max": 30000},
    }

    for event in config["events"]:
        if event == "crash":
            # Simple crash event configuration
            config["crash_event_metrics"]["total_crashes"] = {}
        elif isinstance(event, str) and event == "pageload":
            # Pageload with default metrics
            config["pageload_event_metrics"] = default_pageload_metrics.copy()
        elif isinstance(event, dict) and "pageload" in event:
            # Pageload with custom metrics
            pageload_config = event["pageload"]
            if pageload_config:
                config["pageload_event_metrics"] = pageload_config.copy()
            else:
                config["pageload_event_metrics"] = default_pageload_metrics.copy()
        elif isinstance(event, dict):
            # Handle other event types in the future
            for event_type, event_config in event.items():
                if event_type == "pageload":
                    if event_config:
                        config["pageload_event_metrics"] = event_config.copy()
                    else:
                        config["pageload_event_metrics"] = (
                            default_pageload_metrics.copy()
                        )

    # Now annotate the parsed events using existing functions
    annotatePageloadEventMetrics(config, probeIndex)
    annotateCrashEventMetrics(config, probeIndex)


def annotateHistograms(config: Dict[str, Any], probeIndex: Dict[str, Any]) -> None:
    """Annotate histograms with schema information from probe index.

    Args:
      config: Configuration dictionary containing histograms
      probeIndex: Probe index dictionary containing schema information
    """
    if "histograms" not in config:
        config["histograms"] = {}
        return

    histograms_input = config["histograms"]

    # Normalize histogram input to a dict mapping histogram names to properties
    histogram_props = {}
    if isinstance(histograms_input, list):
        for item in histograms_input:
            if isinstance(item, str):
                histogram_props[item] = {}
            elif isinstance(item, dict):
                # Item is like {"metric.name": {"aggregate": "sum"}}
                for name, props in item.items():
                    histogram_props[name] = props if props else {}
    elif isinstance(histograms_input, dict):
        histogram_props = histograms_input

    config["histograms"] = {}
    for hist in histogram_props:
        config["histograms"][hist] = {}
        if isinstance(histogram_props[hist], dict):
            config["histograms"][hist].update(histogram_props[hist])

        hist_name = hist.split(".")[-1]

        # Prioritize Glean probes for metrics that start with "metrics."
        if hist.startswith("metrics.") and hist_name in probeIndex["glean"]:
            schema = probeIndex["glean"][hist_name]
            config["histograms"][hist]["glean"] = True
            config["histograms"][hist]["desc"] = schema["description"]

            # Mark if the probe is available on desktop or mobile.
            config["histograms"][hist]["available_on_desktop"] = False
            config["histograms"][hist]["available_on_android"] = False

            if "gecko" in schema["repos"]:
                config["histograms"][hist]["available_on_desktop"] = True
                config["histograms"][hist]["available_on_android"] = True
            elif "fenix" in schema["repos"]:
                config["histograms"][hist]["available_on_android"] = True
            elif "desktop" in schema["repos"]:
                config["histograms"][hist]["available_on_desktop"] = True

            # Support timing, memory, and custom distribution types.
            if schema["type"] in [
                "timing_distribution",
                "memory_distribution",
                "custom_distribution",
            ]:
                config["histograms"][hist]["kind"] = "numerical"
                config["histograms"][hist]["distribution_type"] = schema["type"]

                # Copy memory unit for memory distributions
                if schema["type"] == "memory_distribution" and "memory_unit" in schema:
                    config["histograms"][hist]["memory_unit"] = schema["memory_unit"]
            elif schema["type"] == "labeled_counter":
                config["histograms"][hist]["distribution_type"] = schema["type"]
                if "labels" in schema:
                    config["histograms"][hist]["labels"] = schema["labels"]

                # Detect unit: check schema first, then infer from metric name
                if "unit" in schema:
                    config["histograms"][hist]["unit"] = schema["unit"]
                elif hist.endswith("_ms"):
                    config["histograms"][hist]["unit"] = "ms"
                elif hist.endswith("_s"):
                    config["histograms"][hist]["unit"] = "s"

                # Require aggregate field for labeled_counter metrics
                if "aggregate" not in config["histograms"][hist]:
                    print(f"ERROR: labeled_counter metric '{hist}' requires an 'aggregate' field.")
                    print(f"  Specify 'aggregate: sum' (totals) or 'aggregate: percentiles' (median/p75/p95).")
                    print(f"  Example in config:")
                    print(f"    histograms:")
                    print(f"      - {hist}:")
                    print(f"          aggregate: sum  # or 'histogram'")
                    sys.exit(1)

                aggregate = config["histograms"][hist]["aggregate"]
                if aggregate not in ["sum", "percentiles"]:
                    print(f"ERROR: Invalid aggregate value '{aggregate}' for '{hist}'.")
                    print(f"  Must be 'sum' or 'percentiles'.")
                    sys.exit(1)

                if aggregate == "sum":
                    config["histograms"][hist]["kind"] = "categorical"
                else:
                    # percentiles mode: calculate median, p75, p95 per label
                    config["histograms"][hist]["kind"] = "labeled_percentiles"
            else:
                type = schema["type"]
                print(f"ERROR: Type {type} for {hist_name} not currently supported.")
                sys.exit(1)

            # Legacy mirror bounds are available but NOT applied by default.
            # Glean histograms use their native bucket ranges which are often much wider
            # than legacy limits (e.g., memory_total: 16GB legacy vs 115TB Glean).
            # To explicitly use legacy max, set it in the config like:
            #   histograms:
            #     metrics.some_metric:
            #       max: <value>
            # To explicitly disable any max, use max: null
            if "max" in config["histograms"][hist]:
                if config["histograms"][hist]["max"] is None:
                    # Explicitly set to null - remove the key to disable max bucketing
                    del config["histograms"][hist]["max"]
                # Otherwise keep the explicitly configured max value

        # Annotate legacy probe.
        elif hist_name.upper() in probeIndex["legacy"]:
            schema = probeIndex["legacy"][hist_name.upper()]
            config["histograms"][hist]["glean"] = False
            config["histograms"][hist]["desc"] = schema["description"]
            config["histograms"][hist]["available_on_desktop"] = True
            config["histograms"][hist]["available_on_android"] = False
            kind = schema["details"]["kind"]
            print(hist, kind)
            if kind == "categorical" or kind == "boolean" or kind == "enumerated":
                config["histograms"][hist]["kind"] = "categorical"
                if "labels" in schema["details"]:
                    config["histograms"][hist]["labels"] = schema["details"]["labels"]
                elif kind == "boolean":
                    config["histograms"][hist]["labels"] = ["no", "yes"]
                elif "n_buckets" in schema["details"]:
                    n_buckets = schema["details"]["n_buckets"]
                    config["histograms"][hist]["labels"] = list(range(0, n_buckets))
            else:
                config["histograms"][hist]["kind"] = "numerical"

        # Annotate glean probe.
        elif hist_name in probeIndex["glean"]:
            schema = probeIndex["glean"][hist_name]
            config["histograms"][hist]["glean"] = True
            config["histograms"][hist]["desc"] = schema["description"]

            # Mark if the probe is available on desktop or mobile.
            config["histograms"][hist]["available_on_desktop"] = False
            config["histograms"][hist]["available_on_android"] = False

            if "gecko" in schema["repos"]:
                config["histograms"][hist]["available_on_desktop"] = True
                config["histograms"][hist]["available_on_android"] = True
            elif "fenix" in schema["repos"]:
                config["histograms"][hist]["available_on_android"] = True
            elif "desktop" in schema["repos"]:
                config["histograms"][hist]["available_on_desktop"] = True

            # Support timing, memory, and custom distribution types.
            if schema["type"] in [
                "timing_distribution",
                "memory_distribution",
                "custom_distribution",
            ]:
                config["histograms"][hist]["kind"] = "numerical"
                config["histograms"][hist]["distribution_type"] = schema["type"]

                # Copy memory unit for memory distributions
                if schema["type"] == "memory_distribution" and "memory_unit" in schema:
                    config["histograms"][hist]["memory_unit"] = schema["memory_unit"]
            elif schema["type"] == "labeled_counter":
                config["histograms"][hist]["distribution_type"] = schema["type"]
                if "labels" in schema:
                    config["histograms"][hist]["labels"] = schema["labels"]

                # Detect unit: check schema first, then infer from metric name
                if "unit" in schema:
                    config["histograms"][hist]["unit"] = schema["unit"]
                elif hist.endswith("_ms"):
                    config["histograms"][hist]["unit"] = "ms"
                elif hist.endswith("_s"):
                    config["histograms"][hist]["unit"] = "s"

                # Require aggregate field for labeled_counter metrics
                if "aggregate" not in config["histograms"][hist]:
                    print(f"ERROR: labeled_counter metric '{hist}' requires an 'aggregate' field.")
                    print(f"  Specify 'aggregate: sum' (totals) or 'aggregate: percentiles' (median/p75/p95).")
                    print(f"  Example in config:")
                    print(f"    histograms:")
                    print(f"      - {hist}:")
                    print(f"          aggregate: sum  # or 'histogram'")
                    sys.exit(1)

                aggregate = config["histograms"][hist]["aggregate"]
                if aggregate not in ["sum", "percentiles"]:
                    print(f"ERROR: Invalid aggregate value '{aggregate}' for '{hist}'.")
                    print(f"  Must be 'sum' or 'percentiles'.")
                    sys.exit(1)

                if aggregate == "sum":
                    config["histograms"][hist]["kind"] = "categorical"
                else:
                    # percentiles mode: calculate median, p75, p95 per label
                    config["histograms"][hist]["kind"] = "labeled_percentiles"
            else:
                type = schema["type"]
                print(f"ERROR: Type {type} for {hist_name} not currently supported.")
                sys.exit(1)

            # Legacy mirror bounds are available but NOT applied by default.
            # Glean histograms use their native bucket ranges which are often much wider
            # than legacy limits (e.g., memory_total: 16GB legacy vs 115TB Glean).
            # To explicitly use legacy max, set it in the config like:
            #   histograms:
            #     metrics.some_metric:
            #       max: <value>
            # To explicitly disable any max, use max: null
            if "max" in config["histograms"][hist]:
                if config["histograms"][hist]["max"] is None:
                    # Explicitly set to null - remove the key to disable max bucketing
                    del config["histograms"][hist]["max"]
                # Otherwise keep the explicitly configured max value

        else:
            print(f"ERROR: {hist_name} not found in histograms schema.")
            sys.exit(1)


def retrieveNimbusAPI(dataDir: str, slug: str, skipCache: bool) -> Dict[str, Any]:
    """Retrieve experiment data from Nimbus API with caching support.

    Args:
      dataDir: Directory to store cached API responses
      slug: Experiment slug identifier
      skipCache: If True, bypass cache and fetch fresh data

    Returns:
      Dictionary containing experiment data from Nimbus API

    Raises:
      SystemExit: If API request fails after retries
    """
    filename = f"{dataDir}/{slug}-nimbus-API.json"
    if skipCache:
        values = None
    else:
        values = checkForLocalFile(filename)
    if values is not None:
        print(f"Using local config found in {filename}")
        return values

    url = f"https://experimenter.services.mozilla.com/api/v8/experiments/{slug}/"
    print(f"Loading nimbus API from {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        values = response.json()
        with open(filename, "w") as f:
            json.dump(values, f, indent=2)
        return values
    except requests.exceptions.Timeout:
        print(f"Failed to retrieve {url}: Request timed out after 30 seconds")
        sys.exit(1)
    except requests.exceptions.HTTPError:
        print(f"Failed to retrieve {url}: HTTP {response.status_code}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}: {str(e)}")
        sys.exit(1)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to parse response from {url}: {str(e)}")
        sys.exit(1)


# We only care about a few values from the API.
# Specifically, the branch slugs, channels (prioritized) and start/end dates.
def extractValuesFromAPI(api: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant values from Nimbus API response.

    Args:
      api: Raw API response from Nimbus

    Returns:
      Dictionary with extracted experiment metadata
    """
    values = {}
    values["startDate"] = api["startDate"]
    values["endDate"] = api["endDate"]

    # Some experiments can use multiple channels, so select
    # the channel based on the following priority: release > beta > nightly
    if "release" in api["channels"]:
        channel = "release"
    elif "beta" in api["channels"]:
        channel = "beta"
    elif "nightly" in api["channels"]:
        channel = "nightly"
    else:
        available_channels = ", ".join(api.get("channels", []))
        raise ValueError(
            f"No supported channel found. Available channels: [{available_channels}]."
        )

    values["channel"] = channel
    values["isRollout"] = api["isRollout"]

    if values["endDate"] is None:
        now = datetime.datetime.now()
        values["endDate"] = now.strftime("%Y-%m-%d")

    values["branches"] = []

    # If a referenceBranch is defined, or a branch named "control"
    # exists, set it as the first element.
    controlBranch = None
    if "targeting" in api and "referenceBranch" in api["targeting"]:
        controlBranch = values["targeting"]["referenceBranch"]
    elif any(branch["slug"] == "control" for branch in api["branches"]):
        controlBranch = "control"

    if controlBranch:
        values["branches"].append({"name": controlBranch})

    for branch in api["branches"]:
        if branch["slug"] == controlBranch:
            continue
        values["branches"].append({"name": branch["slug"]})

    return values


def parseNimbusAPI(dataDir: str, slug: str, skipCache: bool) -> Dict[str, Any]:
    """Parse experiment data from Nimbus API.

    Args:
      dataDir: Directory to store cached API responses
      slug: Experiment slug identifier
      skipCache: If True, bypass cache and fetch fresh data

    Returns:
      Dictionary with parsed experiment metadata
    """
    apiValues = retrieveNimbusAPI(dataDir, slug, skipCache)
    return extractValuesFromAPI(apiValues)


def promptForMaxDuration(config: Dict[str, Any], args: Any) -> None:
    """Prompt user to limit data collection for long-running release experiments.

    Args:
      config: Configuration dictionary with startDate, endDate, and channel
      args: Arguments object that may contain max_duration_days
    """
    # Skip if max_duration_days is already set
    if hasattr(args, "max_duration_days") and args.max_duration_days is not None:
        return

    # Skip if not a release channel
    channel = config.get("channel", "").lower()
    if channel != "release":
        return

    # Skip if we don't have dates
    if "startDate" not in config or "endDate" not in config:
        return

    # Calculate duration
    try:
        start_date = datetime.datetime.strptime(config["startDate"], "%Y-%m-%d")
        end_date = datetime.datetime.strptime(config["endDate"], "%Y-%m-%d")
        duration_days = (end_date - start_date).days
    except (ValueError, TypeError):
        return

    # Prompt if duration is greater than 7 days
    if duration_days > 7:
        print(
            f"\n⚠️  WARNING: This is a release channel experiment with a {duration_days}-day duration."
        )
        print(f"   Date range: {config['startDate']} to {config['endDate']}")
        print(
            "   Collecting data for the full duration may take a long time and incur high BigQuery costs."
        )
        print("   (You can also set a custom duration using --max-duration-days=N)")
        print()

        response = (
            input(
                "   Would you like to restrict data collection to the last 7 days? [Y/n]: "
            )
            .strip()
            .lower()
        )

        if response in ["", "y", "yes"]:
            args.max_duration_days = 7
            print("   ✓ Data collection will be limited to the last 7 days.")
        else:
            args.max_duration_days = None
            print(
                f"   ✓ Data collection will proceed for the full {duration_days}-day duration."
            )
        print()


def applyMaxDuration(config: Dict[str, Any], max_duration_days: Optional[int]) -> None:
    """Adjust the start date if max_duration_days is specified.

    Args:
      config: Configuration dictionary with startDate and endDate
      max_duration_days: Maximum number of days to collect data (from end date backwards)
    """
    if max_duration_days is None:
        return

    if "endDate" not in config or "startDate" not in config:
        return

    end_date = datetime.datetime.strptime(config["endDate"], "%Y-%m-%d")
    original_start_date = datetime.datetime.strptime(config["startDate"], "%Y-%m-%d")

    # Calculate the new start date (max_duration_days before end date)
    adjusted_start_date = end_date - datetime.timedelta(days=max_duration_days)

    # Only adjust if the new start date is later than the original
    if adjusted_start_date > original_start_date:
        config["startDate"] = adjusted_start_date.strftime("%Y-%m-%d")
        print(
            f"Limiting data collection to last {max_duration_days} days: {config['startDate']} to {config['endDate']}"
        )


def parseConfigFile(configFile: str) -> Dict[str, Any]:
    """Parse YAML config file and add experiment metadata.

    Args:
      configFile: Path to the YAML configuration file

    Returns:
      Dictionary containing the parsed configuration with is_experiment flag
    """
    with open(configFile, "r") as configData:
        config = yaml.safe_load(configData)

    if "branches" in config:
        config["is_experiment"] = False
        # Validate branch names don't start with digits
        for branch in config["branches"]:
            if "name" in branch:
                name = branch["name"]
                if name and name[0].isdigit():
                    print(
                        f"ERROR: Branch name '{name}' cannot start with a digit (SQL identifier restriction)."
                    )
                    print(
                        "Please rename the branch to start with a letter or underscore."
                    )
                    sys.exit(1)
    else:
        config["is_experiment"] = True

    # Handle custom branch conditions if present
    if "prerequisite_cte" in config:
        config["prerequisite_ctes"] = config["prerequisite_cte"]

    # Mark if branches have custom conditions
    has_custom_branches = any(
        "custom_condition" in branch for branch in config.get("branches", [])
    )
    if has_custom_branches:
        config["has_custom_branches"] = True

    return config
