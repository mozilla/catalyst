#!/usr/bin/env python3
import json
import os
import time
import numpy as np
import django
from django.apps import apps
import lib.parser as parser
from django.conf import settings
from lib.telemetry import TelemetryClient
from lib.analysis import DataAnalyzer
from lib.report import ReportGenerator


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def setupDjango():
    if apps.ready:
        return

    TEMPLATES = [
        {
            "BACKEND": "django.template.backends.django.DjangoTemplates",
            "DIRS": [
                os.path.join(os.path.dirname(__file__), "templates", "sql"),
                os.path.join(os.path.dirname(__file__), "templates", "html"),
            ],
        }
    ]
    settings.configure(TEMPLATES=TEMPLATES)
    django.setup()


def setupDirs(slug, dataDir, reportDir, generate_report):
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    if not os.path.isdir(os.path.join(dataDir, slug)):
        os.mkdir(os.path.join(dataDir, slug))
    if generate_report:
        if not os.path.isdir(reportDir):
            os.mkdir(reportDir)


def getResultsForExperiment(slug, dataDir, config, skipCache):
    sqlClient = TelemetryClient(dataDir, config, skipCache)
    telemetryData = sqlClient.getResults()

    # Change the branches to a list for easier use during analysis.
    branch_names = []
    for i in range(len(config["branches"])):
        branch_names.append(config["branches"][i]["name"])
    config["branches"] = branch_names

    # Transform telemetry data from section-based to data-type-based format
    transformedData = transformTelemetryDataByType(telemetryData, config)

    analyzer = DataAnalyzer(config)
    results = analyzer.processTelemetryData(transformedData)

    # Save the queries into the results and cache them.
    queriesFile = os.path.join(dataDir, f"{slug}-queries.json")
    if "queries" in telemetryData and telemetryData["queries"]:
        with open(queriesFile, "w") as f:
            json.dump(telemetryData["queries"], f, indent=2, cls=NpEncoder)
    else:
        queries = checkForLocalResults(queriesFile)
        if queries is not None:
            telemetryData["queries"] = queries

    results["queries"] = telemetryData["queries"]
    return results


def checkForLocalResults(resultsFile):
    if os.path.isfile(resultsFile):
        try:
            with open(resultsFile, "r") as f:
                results = json.load(f)
                return results
        except json.JSONDecodeError:
            return None
    return None


def isValidMetricData(data, metric_name, branch, segment, source_section, data_type):
    """Validate that metric data contains actual measurements.

    Args:
        data: The metric data to validate
        metric_name: Name of the metric for warning messages
        branch: Branch name for warning messages
        segment: Segment name for warning messages
        source_section: Source section for warning messages
        data_type: Type of data (numerical, categorical, scalar)

    Returns:
        True if data is valid, False if empty/invalid
    """
    if data_type in ["numerical", "categorical"]:
        bins = data.get("bins", [])
        counts = data.get("counts", [])
        if not bins or not counts or sum(counts) == 0:
            print(
                f"  WARNING: Skipping {branch}/{segment}: {source_section}.{metric_name} - no data available"
            )
            return False
    elif data_type == "scalar":
        scalar_value = data.get("crash_count", data.get("count", None))
        if scalar_value is None:
            print(
                f"  WARNING: Skipping {branch}/{segment}: {source_section}.{metric_name} - no data available"
            )
            return False

    return True


def transformTelemetryDataByType(telemetryData, config):
    """Transform telemetry data from section-based to data-type-based format.

    Args:
        telemetryData: Raw telemetry data organized by sections
        config: Configuration with metric type information

    Returns:
        Transformed data organized by data types: {
            "numerical": [...],
            "categorical": [...],
            "scalar": [...]
        }
    """
    transformedData = {
        "numerical": [],
        "categorical": [],
        "scalar": [],
        "queries": telemetryData.get("queries", {}),  # Preserve queries
    }

    for branch in config["branches"]:
        for segment in config["segments"]:
            # Transform numerical histograms
            for hist in config.get("histograms", {}):
                if config["histograms"][hist]["kind"] == "numerical":
                    # Use full histogram name as key, shortened name for display
                    hist_name = hist.split(".")[-1]
                    if hist in telemetryData.get(branch, {}).get(segment, {}).get(
                        "histograms", {}
                    ):
                        data = telemetryData[branch][segment]["histograms"][hist]
                        if isValidMetricData(
                            data, hist_name, branch, segment, "histograms", "numerical"
                        ):
                            transformedData["numerical"].append(
                                {
                                    "branch": branch,
                                    "segment": segment,
                                    "metric_name": hist_name,
                                    "source_section": "histograms",
                                    "source_key": hist,
                                    "config": config["histograms"][hist],
                                    "data": data,
                                }
                            )

            # Transform categorical histograms
            for hist in config.get("histograms", {}):
                if config["histograms"][hist]["kind"] == "categorical":
                    # Use full histogram name as key, shortened name for display
                    hist_name = hist.split(".")[-1]
                    if hist in telemetryData.get(branch, {}).get(segment, {}).get(
                        "histograms", {}
                    ):
                        data = telemetryData[branch][segment]["histograms"][hist]
                        if isValidMetricData(
                            data,
                            hist_name,
                            branch,
                            segment,
                            "histograms",
                            "categorical",
                        ):
                            transformedData["categorical"].append(
                                {
                                    "branch": branch,
                                    "segment": segment,
                                    "metric_name": hist_name,
                                    "source_section": "histograms",
                                    "source_key": hist,
                                    "config": config["histograms"][hist],
                                    "data": data,
                                }
                            )

            # Transform pageload event metrics (use annotated kind)
            for metric in config.get("pageload_event_metrics", {}):
                if metric in telemetryData.get(branch, {}).get(segment, {}).get(
                    "pageload_event_metrics", {}
                ):
                    metric_config = config["pageload_event_metrics"][metric]
                    kind = metric_config.get(
                        "kind", "numerical"
                    )  # Default to numerical for backwards compatibility

                    data = telemetryData[branch][segment]["pageload_event_metrics"][
                        metric
                    ]
                    if isValidMetricData(
                        data, metric, branch, segment, "pageload_event_metrics", kind
                    ):
                        transformedData[kind].append(
                            {
                                "branch": branch,
                                "segment": segment,
                                "metric_name": metric,
                                "source_section": "pageload_event_metrics",
                                "source_key": metric,
                                "config": metric_config,
                                "data": data,
                            }
                        )

            # Transform crash event metrics (use annotated kind)
            for metric in config.get("crash_event_metrics", {}):
                if metric in telemetryData.get(branch, {}).get(segment, {}).get(
                    "crash_event_metrics", {}
                ):
                    metric_config = config["crash_event_metrics"][metric]
                    kind = metric_config.get(
                        "kind", "scalar"
                    )  # Default to scalar for backwards compatibility

                    data = telemetryData[branch][segment]["crash_event_metrics"][metric]
                    if isValidMetricData(
                        data, metric, branch, segment, "crash_event_metrics", kind
                    ):
                        transformedData[kind].append(
                            {
                                "branch": branch,
                                "segment": segment,
                                "metric_name": metric,
                                "source_section": "crash_event_metrics",
                                "source_key": metric,
                                "config": metric_config,
                                "data": data,
                            }
                        )

    return transformedData


def generate_report(args):
    startTime = time.time()

    setupDjango()

    # Parse config file.
    print("Loading config file: ", args.config)
    config = parser.parseConfigFile(args.config)
    slug = config["slug"]

    # Setup local dirs
    print("Setting up local directories.")
    setupDirs(slug, args.dataDir, args.reportDir, args.html_report)
    dataDir = os.path.join(args.dataDir, slug)
    reportDir = args.reportDir
    skipCache = args.skip_cache

    # Check for local results first.
    resultsFile = os.path.join(dataDir, f"{slug}-results.json")
    if skipCache:
        results = None
    else:
        results = checkForLocalResults(resultsFile)

    # If results not found, generate them.
    if results is None:
        # Annotate metrics
        parser.annotateMetrics(config)

        if config["is_experiment"]:
            # Parse Nimbus API.
            api = parser.parseNimbusAPI(dataDir, slug, skipCache)

            # Preserve startDate and endDate from config if they exist
            config_start_date = config.get("startDate")
            config_end_date = config.get("endDate")

            # Convert date objects to string format if needed
            if config_start_date is not None and hasattr(config_start_date, "strftime"):
                config_start_date = config_start_date.strftime("%Y-%m-%d")
            if config_end_date is not None and hasattr(config_end_date, "strftime"):
                config_end_date = config_end_date.strftime("%Y-%m-%d")

            config = config | api

            # Override API dates with config dates if provided
            if config_start_date is not None:
                config["startDate"] = config_start_date
            if config_end_date is not None:
                config["endDate"] = config_end_date

            # Prompt user for max duration if needed
            parser.promptForMaxDuration(config, args)

            # Apply max duration limit if specified
            if hasattr(args, "max_duration_days"):
                parser.applyMaxDuration(config, args.max_duration_days)

            # If the experiment is a rollout, then use the non-enrolled branch
            # as the control.
            if config["isRollout"]:
                config["include_non_enrolled_branch"] = True

            # If non-enrolled branch was included, add an extra branch.
            if "include_non_enrolled_branch" in config:
                include_non_enrolled_branch = config["include_non_enrolled_branch"]
                if include_non_enrolled_branch is True or (
                    isinstance(include_non_enrolled_branch, str)
                    and include_non_enrolled_branch.lower() == "true"
                ):
                    config["include_non_enrolled_branch"] = True
                    if config["isRollout"]:
                        config["branches"].insert(0, {"name": "default"})
                    else:
                        config["branches"].append({"name": "default"})
            else:
                config["include_non_enrolled_branch"] = False
        else:
            # For non-experiments, apply max duration if specified
            if hasattr(args, "max_duration_days"):
                parser.applyMaxDuration(config, args.max_duration_days)

        print("Using Config:")
        configStr = json.dumps(config, indent=2)
        print(configStr)

        # Get statistical results
        origConfig = config.copy()
        results = getResultsForExperiment(slug, dataDir, config, skipCache)
        results = results | config
        results["input"] = origConfig

        # Save results to disk.
        print("---------------------------------")
        print(f"Writing results to {resultsFile}")
        with open(resultsFile, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
    else:
        print("---------------------------------")
        print(f"Found local results in {resultsFile}")

    if args.html_report:
        reportFile = os.path.join(reportDir, f"{slug}.html")
        print(f"Generating html report in {reportFile}")

        gen = ReportGenerator(results)
        report = gen.createHTMLReport()
        with open(reportFile, "w") as f:
            f.write(report)

    executionTime = time.time() - startTime
    print(f"Execution time: {executionTime:.1f} seconds")
