"""
BigQuery telemetry data analysis module.

This module provides functionality to generate BigQuery SQL queries,
execute them, and process histogram and pageload event metrics data.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
import pandas as pd
from google.cloud import bigquery
from django.template.loader import get_template


def invalidDataSet(
    df: pd.DataFrame,
    histogram: str,
    branches: List[Dict[str, Any]],
    segments: List[str],
    min_sample_size: int = 1000,
) -> bool:
    """
    Check if entire dataset is invalid (completely empty).

    Args:
        df: DataFrame containing telemetry data
        histogram: Name of the histogram being validated
        branches: List of branch configurations
        segments: List of segment names
        min_sample_size: Minimum required sample size per branch/segment

    Returns:
        True if dataset is completely invalid (empty), False otherwise
    """
    if df.empty:
        print(f"Empty dataset found, removing: {histogram}.")
        return True

    # Check if all branches are empty
    all_branches_empty = True
    for branch in branches:
        branch_name = branch["name"]
        branch_df = df[df["branch"] == branch_name]
        if not branch_df.empty:
            all_branches_empty = False
            break

    if all_branches_empty:
        print(f"All branches empty, removing: {histogram}.")
        return True

    return False


def getInvalidBranchSegments(
    df: pd.DataFrame,
    histogram: str,
    branches: List[Dict[str, Any]],
    segments: List[str],
    min_sample_size: int = 1000,
) -> List[tuple]:
    """
    Identify which specific branch/segment combinations have invalid data.

    Args:
        df: DataFrame containing telemetry data
        histogram: Name of the histogram being validated
        branches: List of branch configurations
        segments: List of segment names
        min_sample_size: Minimum required sample size per branch/segment

    Returns:
        List of (branch_name, segment) tuples that should be excluded
    """
    invalid_combinations = []

    for branch in branches:
        branch_name = branch["name"]
        branch_df = df[df["branch"] == branch_name]

        if branch_df.empty:
            # Entire branch is empty - mark all segments as invalid
            for segment in segments:
                if segment != "All":
                    invalid_combinations.append((branch_name, segment))
            continue

        for segment in segments:
            if segment == "All":
                continue

            branch_segment_df = branch_df[branch_df["segment"] == segment]

            if branch_segment_df.empty:
                print(
                    f"  WARNING: Skipping {branch_name}/{segment}: {histogram} - no data available"
                )
                invalid_combinations.append((branch_name, segment))
                continue

            # Check minimum sample size for histograms and pageload events
            if "counts" in branch_segment_df.columns:
                total_samples = branch_segment_df["counts"].sum()
                if total_samples < min_sample_size:
                    print(
                        f"  WARNING: Skipping {branch_name}/{segment}: {histogram} - "
                        f"insufficient sample size (found {total_samples}, need {min_sample_size})"
                    )
                    invalid_combinations.append((branch_name, segment))
            # For crash events, we don't enforce minimum sample size since 0 crashes is meaningful

    return invalid_combinations


def segments_are_all_OS(segments: List[str]) -> bool:
    """
    Check if all segments are OS-based (allows for faster queries).

    Args:
        segments: List of segment names to validate

    Returns:
        True if all segments are OS-based, False otherwise
    """
    os_segments = set(["Windows", "All", "Linux", "Mac", "Android"])
    for segment in segments:
        if segment not in os_segments:
            return False
    return True


def clean_sql_query(query: str) -> str:
    """
    Clean SQL query by removing empty lines.

    Args:
        query: Raw SQL query string

    Returns:
        Cleaned SQL query with empty lines removed
    """
    return "".join([s for s in query.strip().splitlines(True) if s.strip()])


class TelemetryClient:
    """
    BigQuery client for telemetry data analysis.

    This class handles SQL query generation, BigQuery execution,
    and local caching of results.
    """

    def __init__(self, dataDir: str, config: Dict[str, Any], skipCache: bool = False):
        self.client = bigquery.Client()
        self.config = config
        self.dataDir = dataDir
        self.skipCache = skipCache
        self.queries: List[Dict[str, str]] = []

        # Configure parallel query execution
        # Default to 4 threads - good balance between performance and resource usage
        self.max_workers = config.get("max_parallel_queries", 4)

        # Ensure data directory exists
        os.makedirs(dataDir, exist_ok=True)

    def _executeQueriesInParallel(self, is_experiment: bool = True):
        """
        Execute telemetry queries in parallel for all metric types.

        Args:
            is_experiment: If True, use experiment methods, otherwise use non-experiment methods

        Returns:
            Tuple of (event_metrics, crash_metrics, histograms, invalid_combinations)
        """
        # Track invalid branch/segment combinations for each metric
        invalid_combinations = {}

        # Prepare all queries to run in parallel
        query_tasks = []

        # Select appropriate methods based on experiment type
        if is_experiment:
            pageload_method = self.getPageloadEventData
            crash_method = self.getCrashEventData

            def histogram_method(histogram_name):
                return self.getHistogramData(self.config, histogram_name)
        else:
            pageload_method = self.getPageloadEventDataNonExperiment
            crash_method = self.getCrashEventDataNonExperiment

            def histogram_method(histogram_name):
                return self.getHistogramDataNonExperiment(self.config, histogram_name)

        # Add pageload event metric queries
        for metric in self.config["pageload_event_metrics"]:
            query_tasks.append({
                'type': 'pageload',
                'metric': metric,
                'method': pageload_method
            })

        # Add crash event metric queries
        for metric in self.config["crash_event_metrics"]:
            query_tasks.append({
                'type': 'crash',
                'metric': metric,
                'method': crash_method
            })

        # Add histogram queries
        for histogram in self.config["histograms"]:
            query_tasks.append({
                'type': 'histogram',
                'metric': histogram,
                'method': lambda h=histogram: histogram_method(h)
            })

        # Execute all queries in parallel
        event_metrics = {}
        crash_metrics = {}
        histograms = {}

        print(f"Running {len(query_tasks)} queries in parallel using {self.max_workers} threads...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in query_tasks:
                if task['type'] == 'histogram':
                    future = executor.submit(task['method'])
                else:
                    future = executor.submit(task['method'], task['metric'])
                future_to_task[future] = task

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    df = future.result()
                    print(f"Completed {task['type']} query for {task['metric']}")
                    print(df)

                    # Process based on metric type
                    if task['type'] == 'pageload':
                        metric = task['metric']
                        if not invalidDataSet(
                            df,
                            f"pageload event: {metric}",
                            self.config["branches"],
                            self.config["segments"],
                        ):
                            event_metrics[metric] = df
                            invalid_combinations[f"pageload_{metric}"] = getInvalidBranchSegments(
                                df,
                                f"pageload event: {metric}",
                                self.config["branches"],
                                self.config["segments"],
                            )

                    elif task['type'] == 'crash':
                        metric = task['metric']
                        if not invalidDataSet(
                            df,
                            f"crash event: {metric}",
                            self.config["branches"],
                            self.config["segments"],
                        ):
                            crash_metrics[metric] = df
                            invalid_combinations[f"crash_{metric}"] = getInvalidBranchSegments(
                                df,
                                f"crash event: {metric}",
                                self.config["branches"],
                                self.config["segments"],
                            )

                    elif task['type'] == 'histogram':
                        histogram = task['metric']
                        if not invalidDataSet(
                            df, histogram, self.config["branches"], self.config["segments"]
                        ):
                            histograms[histogram] = df
                            invalid_combinations[f"histogram_{histogram}"] = getInvalidBranchSegments(
                                df, histogram, self.config["branches"], self.config["segments"]
                            )

                except Exception as e:
                    print(f"Error processing {task['type']} query for {task['metric']}: {e}")

        print(f"Parallel query execution completed. Results: {len(event_metrics)} pageload, {len(crash_metrics)} crash, {len(histograms)} histogram metrics")

        return event_metrics, crash_metrics, histograms, invalid_combinations

    def collectResultsFromQuery_OS_segments(
        self,
        results: Dict[str, Any],
        branch: str,
        segment: str,
        event_metrics: Dict[str, pd.DataFrame],
        histograms: Dict[str, pd.DataFrame],
        crash_metrics: Dict[str, pd.DataFrame],
        invalid_combinations: Dict[str, List[tuple]] = None,
    ) -> None:
        if invalid_combinations is None:
            invalid_combinations = {}

        for histogram in self.config["histograms"]:
            # Skip this branch/segment if it's invalid for this histogram
            histogram_key = f"histogram_{histogram}"
            if histogram_key in invalid_combinations:
                if (branch, segment) in invalid_combinations[histogram_key]:
                    continue

            df = histograms[histogram]
            if segment == "All":
                subset = (
                    df[df["branch"] == branch][["bucket", "counts"]]
                    .groupby(["bucket"])
                    .sum()
                )
                buckets = list(subset.index)
                counts = list(subset["counts"])
            else:
                subset = df[(df["segment"] == segment) & (df["branch"] == branch)]
                buckets = list(subset["bucket"])
                counts = list(subset["counts"])

            # Some clients report bucket sizes that are not real, and these buckets
            # end up having 1-5 samples in them.  Filter these out entirely.
            if self.config["histograms"][histogram]["kind"] == "numerical":
                remove = []
                for i in range(1, len(counts) - 1):
                    if (counts[i - 1] > 1000 and counts[i] < counts[i - 1] / 100) or (
                        counts[i + 1] > 1000 and counts[i] < counts[i + 1] / 100
                    ):
                        remove.append(i)
                for i in sorted(remove, reverse=True):
                    del buckets[i]
                    del counts[i]

            # Add labels to the buckets for categorical histograms.
            if self.config["histograms"][histogram]["kind"] == "categorical":
                labels = self.config["histograms"][histogram]["labels"]

                # Remove overflow bucket if it exists
                if len(labels) == (len(buckets) - 1) and counts[-1] == 0:
                    del buckets[-1]
                    del counts[-1]

                # Add missing buckets so they line up in each branch.
                if len(labels) > len(buckets):
                    for i in range(len(buckets)):
                        print(buckets[i], counts[i])
                    new_counts = []
                    for i, b in enumerate(labels):
                        j = buckets.index(b) if b in buckets else None
                        if j:
                            new_counts.append(counts[j])
                        else:
                            new_counts.append(0)
                    counts = new_counts

                # Remap bucket values to the appropriate label names.
                buckets = labels

            # If there is a max, then overflow larger buckets into the max.
            if "max" in self.config["histograms"][histogram]:
                maxBucket = self.config["histograms"][histogram]["max"]
                remove = []
                maxBucketCount = 0
                for i, x in enumerate(buckets):
                    if x >= maxBucket:
                        remove.append(i)
                        maxBucketCount = maxBucketCount + counts[i]
                for i in sorted(remove, reverse=True):
                    del buckets[i]
                    del counts[i]
                buckets.append(maxBucket)
                counts.append(maxBucketCount)

            assert len(buckets) == len(counts)
            results[branch][segment]["histograms"][histogram] = {}
            results[branch][segment]["histograms"][histogram]["bins"] = buckets
            results[branch][segment]["histograms"][histogram]["counts"] = counts
            print(f"    segment={segment} len(histogram: {histogram}) = ", len(buckets))

        for metric in self.config["pageload_event_metrics"]:
            # Skip this branch/segment if it's invalid for this pageload metric
            pageload_key = f"pageload_{metric}"
            if pageload_key in invalid_combinations:
                if (branch, segment) in invalid_combinations[pageload_key]:
                    continue

            df = event_metrics[metric]
            if segment == "All":
                subset = (
                    df[df["branch"] == branch][["bucket", "counts"]]
                    .groupby(["bucket"])
                    .sum()
                )
                buckets = list(subset.index)
                counts = list(subset["counts"])
            else:
                subset = df[(df["segment"] == segment) & (df["branch"] == branch)]
                buckets = list(subset["bucket"])
                counts = list(subset["counts"])

            assert len(buckets) == len(counts)
            results[branch][segment]["pageload_event_metrics"][metric] = {}
            results[branch][segment]["pageload_event_metrics"][metric]["bins"] = buckets
            results[branch][segment]["pageload_event_metrics"][metric][
                "counts"
            ] = counts
            print(
                f"    segment={segment} len(pageload event: {metric}) = ", len(buckets)
            )

        for metric in self.config["crash_event_metrics"]:
            # Skip this metric if it was filtered out during data loading (no data available)
            if metric not in crash_metrics:
                continue

            # Skip this branch/segment if it's invalid for this crash metric
            crash_key = f"crash_{metric}"
            if crash_key in invalid_combinations:
                if (branch, segment) in invalid_combinations[crash_key]:
                    continue

            df = crash_metrics[metric]
            if segment == "All":
                subset = df[df["branch"] == branch][["crash_count"]].sum()
                crash_count = int(subset["crash_count"])
            else:
                subset = df[(df["segment"] == segment) & (df["branch"] == branch)]
                crash_count = (
                    int(subset["crash_count"].sum()) if not subset.empty else 0
                )

            results[branch][segment]["crash_event_metrics"][metric] = {
                "crash_count": crash_count
            }
            print(
                f"    segment={segment} crash metric: {metric} = {crash_count} crashes"
            )

    def getResults(self) -> Dict[str, Any]:
        """
        Get telemetry results for the configured experiment.

        Returns:
            Dictionary containing processed histogram and pageload event data
        """
        if self.config["is_experiment"] is True:
            return self.getResultsForExperiment()
        else:
            return self.getResultsForNonExperiment()

    def getResultsForNonExperiment(self):
        # Execute all queries in parallel using consolidated method
        event_metrics, crash_metrics, histograms, invalid_combinations = self._executeQueriesInParallel(is_experiment=False)

        # Combine histogram and pageload event results.
        results = {}
        for i in range(len(self.config["branches"])):
            branch_name = self.config["branches"][i]["name"]
            results[branch_name] = {}
            for segment in self.config["segments"]:
                print(
                    f"Aggregating results for segment={segment} "
                    f"and branch={branch_name}"
                )
                results[branch_name][segment] = {
                    "histograms": {},
                    "pageload_event_metrics": {},
                    "crash_event_metrics": {},
                }

                # Special case when segments is OS only.
                self.collectResultsFromQuery_OS_segments(
                    results,
                    branch_name,
                    segment,
                    event_metrics,
                    histograms,
                    crash_metrics,
                    invalid_combinations,
                )

        results["queries"] = self.queries
        return results

    def getResultsForExperiment(self):
        # Execute all queries in parallel using consolidated method
        event_metrics, crash_metrics, histograms, invalid_combinations = self._executeQueriesInParallel(is_experiment=True)

        # Combine histogram and pageload event results.
        results = {}
        for branch in self.config["branches"]:
            branch_name = branch["name"]
            results[branch_name] = {}
            for segment in self.config["segments"]:
                print(
                    f"Aggregating results for segment={segment} "
                    f"and branch={branch_name}"
                )
                results[branch_name][segment] = {
                    "histograms": {},
                    "pageload_event_metrics": {},
                    "crash_event_metrics": {},
                }

                # Special case when segments is OS only.
                self.collectResultsFromQuery_OS_segments(
                    results,
                    branch_name,
                    segment,
                    event_metrics,
                    histograms,
                    crash_metrics,
                    invalid_combinations,
                )

        results["queries"] = self.queries
        return results

    def generatePageloadEventQuery_OS_segments_non_experiment(self, metric):
        t = get_template("other/glean/pageload_events_os_segments.sql")

        minVal = self.config["pageload_event_metrics"][metric]["min"]
        maxVal = self.config["pageload_event_metrics"][metric]["max"]

        branches = self.config["branches"]
        for i in range(len(branches)):
            branches[i]["last"] = False
            if "version" in self.config["branches"][i]:
                version = self.config["branches"][i]["version"]
                branches[i][
                    "ver_condition"
                ] = f"AND SPLIT(client_info.app_display_version, '.')[offset(0)] = \"{version}\""
            if "architecture" in self.config["branches"][i]:
                arch = self.config["branches"][i]["architecture"]
                branches[i][
                    "arch_condition"
                ] = f'AND client_info.architecture = "{arch}"'
            if "glean_conditions" in self.config["branches"][i]:
                branches[i]["glean_conditions"] = self.config["branches"][i][
                    "glean_conditions"
                ]
        branches[-1]["last"] = True

        print(branches)

        context = {
            "minVal": minVal,
            "maxVal": maxVal,
            "metric": metric,
            "branches": branches,
            "pageload_event_filter": self.config.get("pageload_event_filter"),
        }

        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Pageload event: {metric}", "query": query})
        return query

    def generatePageloadEventQuery_OS_segments(self, metric):
        t = get_template("experiment/glean/pageload_events_os_segments.sql")

        print(self.config["pageload_event_metrics"][metric])

        metricMin = self.config["pageload_event_metrics"][metric]["min"]
        metricMax = self.config["pageload_event_metrics"][metric]["max"]

        context = {
            "include_non_enrolled_branch": self.config["include_non_enrolled_branch"],
            "minVal": metricMin,
            "maxVal": metricMax,
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "metric": metric,
            "pageload_event_filter": self.config.get("pageload_event_filter"),
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Pageload event: {metric}", "query": query})
        return query

    # Not currently used, and not well supported.
    def generatePageloadEventQuery_Generic(self):
        t = get_template("archived/events_generic.sql")

        segmentInfo = []
        for segment in self.config["segments"]:
            segmentInfo.append(
                {"name": segment, "conditions": self.config["segments"][segment]}
            )

        maxBucket = 0
        minBucket = 30000
        for metric in self.config["pageload_event_metrics"]:
            metricMin = self.config["pageload_event_metrics"][metric]["min"]
            metricMax = self.config["pageload_event_metrics"][metric]["max"]
            if metricMax > maxBucket:
                maxBucket = metricMax
            if metricMin < minBucket:
                minBucket = metricMin

        context = {
            "minBucket": minBucket,
            "maxBucket": maxBucket,
            "is_experiment": self.config["is_experiment"],
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "metrics": self.config["pageload_event_metrics"],
            "segments": segmentInfo,
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Pageload event: {metric}", "query": query})
        return query

    # Use *_os_segments queries if the segments is OS only which is much faster than generic query.
    def generateHistogramQuery_OS_segments_legacy(self, histogram):
        t = get_template("experiment/legacy/histogram_os_segments.sql")

        context = {
            "include_non_enrolled_branch": self.config["include_non_enrolled_branch"],
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "histogram": histogram,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def generateHistogramQuery_OS_segments_glean(self, histogram):
        t = get_template("experiment/glean/histogram_os_segments.sql")

        context = {
            "include_non_enrolled_branch": self.config["include_non_enrolled_branch"],
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "histogram": histogram,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def generateHistogramQuery_OS_segments_non_experiment_legacy(self, histogram):
        t = get_template("other/legacy/histogram_os_segments.sql")

        branches = self.config["branches"]
        for i in range(len(branches)):
            branches[i]["last"] = False
            if "version" in self.config["branches"][i]:
                version = self.config["branches"][i]["version"]
                branches[i][
                    "ver_condition"
                ] = f"AND SPLIT(application.display_version, '.')[offset(0)] = \"{version}\""
            if "architecture" in self.config["branches"][i]:
                arch = self.config["branches"][i]["architecture"]
                branches[i][
                    "arch_condition"
                ] = f'AND application.architecture = "{arch}"'
            if "legacy_conditions" in self.config["branches"][i]:
                branches[i]["legacy_conditions"] = self.config["branches"][i][
                    "legacy_conditions"
                ]

        branches[-1]["last"] = True

        context = {
            "histogram": histogram,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
            "branches": branches,
            "channel": self.config["branches"][0]["channel"],
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def generateHistogramQuery_OS_segments_non_experiment_glean(self, histogram):
        t = get_template("other/glean/histogram_os_segments.sql")

        branches = self.config["branches"]
        for i in range(len(branches)):
            branches[i]["last"] = False
            if "version" in self.config["branches"][i]:
                version = self.config["branches"][i]["version"]
                branches[i][
                    "ver_condition"
                ] = f"AND SPLIT(client_info.app_display_version, '.')[offset(0)] = \"{version}\""
            if "architecture" in self.config["branches"][i]:
                arch = self.config["branches"][i]["architecture"]
                branches[i][
                    "arch_condition"
                ] = f'AND client_info.architecture = "{arch}"'
            if "glean_conditions" in self.config["branches"][i]:
                branches[i]["glean_conditions"] = self.config["branches"][i][
                    "glean_conditions"
                ]

        branches[-1]["last"] = True

        context = {
            "histogram": histogram,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
            "branches": branches,
        }

        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    # Not currently used, and not well supported.
    def generateHistogramQuery_Generic(self, histogram):
        t = get_template("archived/histogram_generic.sql")

        segmentInfo = []
        for segment in self.config["segments"]:
            segmentInfo.append(
                {"name": segment, "conditions": self.config["segments"][segment]}
            )

        context = {
            "is_experiment": self.config["is_experiment"],
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "histogram": histogram,
            "available_available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
            "segments": segmentInfo,
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def generateCrashEventQuery_OS_segments_non_experiment(self, metric):
        t = get_template("other/glean/crash_events_os_segments.sql")

        branches = self.config["branches"]
        for i in range(len(branches)):
            branches[i]["last"] = False
            if "version" in self.config["branches"][i]:
                version = self.config["branches"][i]["version"]
                branches[i][
                    "ver_condition"
                ] = f"AND SPLIT(client_info.app_display_version, '.')[offset(0)] = \"{version}\""
            if "architecture" in self.config["branches"][i]:
                arch = self.config["branches"][i]["architecture"]
                branches[i][
                    "arch_condition"
                ] = f'AND client_info.architecture = "{arch}"'
            if "glean_conditions" in self.config["branches"][i]:
                branches[i]["glean_conditions"] = self.config["branches"][i][
                    "glean_conditions"
                ]
        branches[-1]["last"] = True

        context = {
            "metric": metric,
            "branches": branches,
        }

        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Crash event: {metric}", "query": query})
        return query

    def generateCrashEventQuery_OS_segments(self, metric):
        t = get_template("experiment/glean/crash_events_os_segments.sql")

        context = {
            "include_non_enrolled_branch": self.config["include_non_enrolled_branch"],
            "slug": self.config["slug"],
            "channel": self.config["channel"],
            "startDate": self.config["startDate"],
            "endDate": self.config["endDate"],
            "metric": metric,
        }
        query = t.render(context)
        query = clean_sql_query(query)
        self.queries.append({"name": f"Crash event: {metric}", "query": query})
        return query

    def checkForExistingData(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Check for existing cached data.

        Args:
            filename: Path to cached file

        Returns:
            DataFrame if cached data exists, None otherwise
        """
        if self.skipCache:
            return None

        try:
            df = pd.read_pickle(filename)
            print(f"Found local data in {filename}")
            return df
        except Exception:
            return None

    def getHistogramDataNonExperiment(
        self, config: Dict[str, Any], histogram: str
    ) -> pd.DataFrame:
        slug = config["slug"]
        hist_name = histogram.split(".")[-1]
        filename = os.path.join(self.dataDir, f"{slug}-{hist_name}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            if config["histograms"][histogram]["glean"]:
                query = self.generateHistogramQuery_OS_segments_non_experiment_glean(
                    histogram
                )
            else:
                query = self.generateHistogramQuery_OS_segments_non_experiment_legacy(
                    histogram
                )
        else:
            print("No current support for generic non-experiment queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' histogram results for {histogram} to disk.")
        df.to_pickle(filename)
        return df

    def getHistogramData(self, config: Dict[str, Any], histogram: str) -> pd.DataFrame:
        slug = config["slug"]
        hist_name = histogram.split(".")[-1]
        filename = os.path.join(self.dataDir, f"{slug}-{hist_name}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            if config["histograms"][histogram]["glean"]:
                query = self.generateHistogramQuery_OS_segments_glean(histogram)
            else:
                query = self.generateHistogramQuery_OS_segments_legacy(histogram)
        else:
            # Generic segments are not well supported right now.
            print("No current support for generic non-experiment queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' histogram results for {histogram} to disk.")
        df.to_pickle(filename)
        return df

    def getPageloadEventDataNonExperiment(self, metric: str) -> pd.DataFrame:
        slug = self.config["slug"]
        filename = os.path.join(self.dataDir, f"{slug}-pageload-events-{metric}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generatePageloadEventQuery_OS_segments_non_experiment(metric)
        else:
            print("Generic non-experiment query currently not supported.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' pageload event results to disk.")
        df.to_pickle(filename)
        return df

    def getPageloadEventData(self, metric: str) -> pd.DataFrame:
        slug = self.config["slug"]
        filename = os.path.join(self.dataDir, f"{slug}-pageload-events-{metric}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generatePageloadEventQuery_OS_segments(metric)
        else:
            # query = self.generatePageloadEventQuery_Generic()
            print("No current support for generic pageload event queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' pageload event results to disk.")
        df.to_pickle(filename)
        return df

    def getCrashEventDataNonExperiment(self, metric: str) -> pd.DataFrame:
        slug = self.config["slug"]
        filename = os.path.join(self.dataDir, f"{slug}-crash-events-{metric}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generateCrashEventQuery_OS_segments_non_experiment(metric)
        else:
            print("Generic non-experiment query currently not supported.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' crash event results to disk.")
        df.to_pickle(filename)
        return df

    def getCrashEventData(self, metric: str) -> pd.DataFrame:
        slug = self.config["slug"]
        filename = os.path.join(self.dataDir, f"{slug}-crash-events-{metric}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generateCrashEventQuery_OS_segments(metric)
        else:
            print("No current support for generic crash event queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()
        print(f"Writing '{slug}' crash event results to disk.")
        df.to_pickle(filename)
        return df
