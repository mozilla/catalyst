import os
import sys
import pandas as pd
from google.cloud import bigquery
from django.template.loader import get_template
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def clean_sql_query(query):
    """
    Clean up SQL query string by removing empty lines and normalizing whitespace.

    Args:
        query (str): The SQL query string to clean

    Returns:
        str: The cleaned SQL query string
    """
    if not query or not query.strip():
        return ""

    # Split into lines, strip whitespace from each line, and filter out empty lines
    lines = [line.strip() for line in query.split("\n")]
    non_empty_lines = [line for line in lines if line]

    # Join with single newlines
    return "\n".join(non_empty_lines)


# Remove any histograms that have empty datasets in
# either a branch, or branch segment.
def invalidDataSet(df, histogram, branches, segments):
    if df.empty:
        print(f"Empty dataset found, removing: {histogram}.")
        return True

    for branch in branches:
        branch_name = branch["name"]
        branch_df = df[df["branch"] == branch_name]
        if branch_df.empty:
            print(
                f"Empty dataset found for branch={branch_name}, removing: {histogram}."
            )
            return True
        for segment in segments:
            if segment == "All":
                continue
            branch_segment_df = branch_df[branch_df["segment"] == segment]
            if branch_segment_df.empty:
                print(
                    f"Empty dataset found for segment={segment}, removing: {histogram}."
                )
                return True

    return False


def getInvalidBranchSegments(df, histogram, branches, segments, min_sample_size=1000):
    invalid_combinations = []

    for branch in branches:
        branch_name = branch["name"]
        branch_df = df[df["branch"] == branch_name]

        if branch_df.empty:
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

            if "counts" in branch_segment_df.columns:
                total_samples = branch_segment_df["counts"].sum()
                if total_samples < min_sample_size:
                    print(
                        f"  WARNING: Skipping {branch_name}/{segment}: {histogram} - "
                        f"insufficient sample size (found {total_samples}, need {min_sample_size})"
                    )
                    invalid_combinations.append((branch_name, segment))

    return invalid_combinations


def segments_are_all_OS(segments):
    os_segments = set(["Windows", "All", "Linux", "Mac", "Android"])
    for segment in segments:
        if segment not in os_segments:
            return False
    return True


def config_has_custom_branches(config):
    """Check if config has custom branch conditions."""
    if config.get("has_custom_branches", False):
        return True
    if "branches" in config:
        for branch in config["branches"]:
            if "custom_condition" in branch:
                return True
    return False


def config_has_custom_segments(config):
    """Check if config has custom segment conditions."""
    if "custom_segments_info" in config:
        return True
    if "segments" in config:
        for segment in config["segments"]:
            if isinstance(segment, dict) and "condition" in segment:
                return True
    return False


class TelemetryClient:
    """
    Client for fetching and processing telemetry data from BigQuery.

    This class handles SQL query generation, BigQuery execution,
    and local caching of results.
    """

    def __init__(self, dataDir: str, config: dict, skipCache: bool = False):
        self.client = bigquery.Client()
        self.config = config
        self.dataDir = dataDir
        self.skipCache = skipCache
        self.queries = []

        # Configure parallel query execution
        # Default to 4 threads - good balance between performance and resource usage
        self.max_workers = config.get("max_parallel_queries", 4)

        # Ensure data directory exists
        os.makedirs(dataDir, exist_ok=True)

        # Check authentication early to provide clear error message
        self._check_authentication()

    def _check_authentication(self):
        """
        Check if BigQuery authentication is properly configured.

        Raises:
            SystemExit: If authentication is not properly configured
        """
        try:
            # Try a simple query to test authentication
            # Use a minimal query that should work if authenticated
            query = "SELECT 1 as test_auth LIMIT 1"
            job = self.client.query(query)
            # Don't wait for results, just check if the job was created
            job.result(max_results=1)
        except Exception as e:
            error_message = str(e).lower()
            if (
                "reauthentication is needed" in error_message
                or "authentication" in error_message
            ):
                print("\n❌ BigQuery Authentication Error")
                print(
                    "Your Google Cloud authentication has expired or is not configured."
                )
                print("\nTo fix this, please run:")
                print("  gcloud auth application-default login")
                print("\nThen try running this command again.")
                sys.exit(1)
            else:
                # Some other error - let it propagate normally
                print(f"\n❌ Error testing BigQuery connection: {e}")
                print("Please check your Google Cloud configuration.")
                sys.exit(1)

    def _get_single_os_filter(self):
        """
        Check if there's a single OS segment configured to optimize queries.

        Returns:
            str or None: The OS name if a single OS segment is configured, None otherwise
        """
        segments = self.config.get("segments", [])
        if len(segments) == 1:
            segment = segments[0]
            # Only add filter for actual OS values, not "All"
            if segment in ["Mac", "Windows", "Linux", "Android"]:
                return segment
        return None

    def _calculateSamplePct(self, histogram):
        """Calculate sampling percentage or LIMIT to cap queries at target entries."""
        sample_pct = self.config.get("sample_pct")

        # Handle auto sampling
        if sample_pct == "auto":
            return self._performAutoSampling(histogram)

        # If manual sampling is set, use it
        if sample_pct:
            return sample_pct

        # Check for per-metric sampling
        metric_sampling = self.config.get("sample_pct_by_metric", {})
        metric_name = histogram.split(".")[-1]  # Extract metric name
        if metric_name in metric_sampling:
            return metric_sampling[metric_name]

        return None  # No sampling

    def _performAutoSampling(self, histogram):
        """Perform automatic sampling by counting entries in the time period."""
        target_entries = self.config.get(
            "sample_pct_auto_target", 500000000
        )  # Default 500M

        print(f"Auto-sampling {histogram}: counting entries in time period...")

        # Get date range for the count query
        if self.config.get("startDate") and self.config.get("endDate"):
            start_date = self.config["startDate"]
            end_date = self.config["endDate"]
        else:
            # Use first branch dates as fallback
            branches = self.config["branches"]
            start_date = branches[0]["startDate"]
            end_date = branches[0]["endDate"]

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        duration_days = (end_dt - start_dt).days + 1

        # Build count query for single day sample
        count_query = self._buildCountQuery(histogram, start_date, end_date)

        print(f"Running count query for {histogram}...")
        import time

        start_time = time.time()
        try:
            job = self.client.query(count_query)
            result = job.to_dataframe()
            query_time = time.time() - start_time
            print(f"Count query completed in {query_time:.1f} seconds")

            if result.empty:
                print(f"No data found for {histogram}")
                return None

            daily_rows = int(result.iloc[0]["daily_rows"])
            daily_entries = int(result.iloc[0]["daily_entries"])
            avg_entries_per_row = daily_entries / daily_rows if daily_rows > 0 else 0

            # Extrapolate to full duration
            total_rows = daily_rows * duration_days
            total_entries = daily_entries * duration_days

            print(
                f"Found {daily_rows:,} daily rows with avg {avg_entries_per_row:.1f} entries/row"
            )
            print(
                f"Estimated {duration_days} days: {total_rows:,} rows = {total_entries:,} total entries for {histogram}"
            )

            if total_entries <= target_entries:
                print(f"No sampling needed ({total_entries:,} ≤ {target_entries:,})")
                return None

            # Calculate required sampling percentage based on rows (since sampling affects row count)
            target_rows = (
                target_entries / avg_entries_per_row
                if avg_entries_per_row > 0
                else target_entries
            )
            required_pct = (target_rows / total_rows) * 100

            # Set minimum sampling of 0.1% (1 in 1,000)
            min_pct = 0.1
            if required_pct < min_pct:
                print(
                    f"Required sampling {required_pct:.6f}% is below minimum {min_pct}%, using {min_pct}% instead"
                )
                required_pct = min_pct

            # For precise sampling, choose modulus based on scale needed
            # Start with 1,000 to support 0.1% minimum (1 in 1,000)
            modulus = 1000
            required_threshold = int((required_pct / 100) * modulus)

            # If we need finer precision, scale up modulus
            if required_threshold == 0:
                modulus = 1000000  # 0.0001% precision
                required_threshold = int((required_pct / 100) * modulus)

                # If still 0, use maximum modulus
                if required_threshold == 0:
                    modulus = 100000000  # 0.000001% precision
                    required_threshold = max(1, int((required_pct / 100) * modulus))

            actual_pct = (required_threshold / modulus) * 100
            estimated_rows = int(total_rows * actual_pct / 100)
            estimated_entries = int(estimated_rows * avg_entries_per_row)

            print(
                f"Using {actual_pct:.6f}% sampling (threshold {required_threshold}/{modulus}) ({total_rows:,} rows → ~{estimated_rows:,} rows = ~{estimated_entries:,} entries)"
            )
            return required_threshold, modulus  # Return threshold and modulus

        except Exception as e:
            print(f"Error counting entries for {histogram}: {e}")
            print("Falling back to 1% sampling")
            return 100, 10000  # 1% sampling as fallback

    def _buildCountQuery(self, histogram, start_date, end_date):
        """Build a count query to estimate entries for a histogram using single day sample."""
        # Use most recent day for sampling
        sample_date = end_date

        return f"""
        WITH desktop_stats AS (
            SELECT
                COUNT(*) as row_count,
                AVG((SELECT SUM(value) FROM UNNEST({histogram}.values))) as avg_entries
            FROM `mozdata.firefox_desktop.metrics` m
            WHERE DATE(submission_timestamp) = DATE('{sample_date}')
              AND {histogram} IS NOT NULL
              AND normalized_channel = 'release'
              AND normalized_app_name = 'Firefox'
        ),
        fenix_stats AS (
            SELECT
                COUNT(*) as row_count,
                AVG((SELECT SUM(value) FROM UNNEST({histogram}.values))) as avg_entries
            FROM `mozdata.fenix.metrics` m
            WHERE DATE(submission_timestamp) = DATE('{sample_date}')
              AND {histogram} IS NOT NULL
        )
        SELECT
            (COALESCE(d.row_count, 0) + COALESCE(f.row_count, 0)) as daily_rows,
            ((COALESCE(d.row_count, 0) * COALESCE(d.avg_entries, 0)) +
             (COALESCE(f.row_count, 0) * COALESCE(f.avg_entries, 0))) as daily_entries
        FROM desktop_stats d, fenix_stats f
        """

    def _parseSamplingConfig(self, sampling_config):
        """Parse sampling configuration - handles both single percentage and (threshold, modulus) tuple."""
        if sampling_config is None:
            return None, None
        elif isinstance(sampling_config, tuple):
            # Auto-sampling returns (threshold, modulus)
            threshold, modulus = sampling_config
            return threshold, modulus
        else:
            # Manual percentage - convert to threshold/modulus format
            return sampling_config, None

    def _processHistogramsInParallel(self, histograms_to_process, is_experiment=True):
        """Process histograms in parallel using ThreadPoolExecutor."""
        if self.max_workers <= 1 or len(histograms_to_process) <= 1:
            # Fall back to sequential processing
            return self._processHistogramsSequential(
                histograms_to_process, is_experiment
            )

        print(
            f"Processing {len(histograms_to_process)} histograms in parallel using {self.max_workers} threads..."
        )

        histograms = {}
        remove = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all histogram processing tasks
            future_to_histogram = {}
            for histogram in histograms_to_process:
                print(f"Starting thread for histogram: {histogram}")
                if is_experiment:
                    future = executor.submit(
                        self.getHistogramData, self.config, histogram
                    )
                else:
                    future = executor.submit(
                        self.getHistogramDataNonExperiment, self.config, histogram
                    )
                future_to_histogram[future] = histogram

            # Collect results as they complete
            for future in as_completed(future_to_histogram):
                histogram = future_to_histogram[future]
                try:
                    df = future.result()
                    print(f"Completed processing histogram: {histogram}")

                    # Check if dataset is valid
                    if invalidDataSet(
                        df, histogram, self.config["branches"], self.config["segments"]
                    ):
                        print(f"Empty dataset found, marking for removal: {histogram}")
                        remove.append(histogram)
                    else:
                        histograms[histogram] = df

                except Exception as e:
                    print(f"Error processing histogram {histogram}: {e}")
                    remove.append(histogram)

        return histograms, remove

    def _processHistogramsSequential(self, histograms_to_process, is_experiment=True):
        """Process histograms sequentially (fallback method)."""
        histograms = {}
        remove = []

        for histogram in histograms_to_process:
            try:
                if is_experiment:
                    df = self.getHistogramData(self.config, histogram)
                else:
                    df = self.getHistogramDataNonExperiment(self.config, histogram)

                # Check if dataset is valid
                if invalidDataSet(
                    df, histogram, self.config["branches"], self.config["segments"]
                ):
                    print(f"Empty dataset found, marking for removal: {histogram}")
                    remove.append(histogram)
                else:
                    histograms[histogram] = df

            except Exception as e:
                print(f"Error processing histogram {histogram}: {e}")
                remove.append(histogram)

        return histograms, remove

    def collectResultsFromQuery_OS_segments(
        self,
        results,
        branch,
        segment,
        event_metrics,
        histograms,
        crash_event_metrics=None,
    ):
        for histogram in self.config["histograms"]:
            # Skip labeled_percentiles metrics - they're processed separately
            if self.config["histograms"][histogram].get("kind") == "labeled_percentiles":
                continue
            # Skip quantity_percentiles metrics - they're processed separately
            if self.config["histograms"][histogram].get("kind") == "quantity_percentiles":
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
                labels = self.config["histograms"][histogram].get("labels")

                if labels:
                    # Remove overflow bucket if it exists
                    if len(labels) == (len(buckets) - 1) and counts[-1] == 0:
                        del buckets[-1]
                        del counts[-1]

                    # Create a bucket->count mapping from query results
                    bucket_to_count = dict(zip(buckets, counts))

                    # Remap counts to match label order, filling missing labels with 0
                    new_counts = []
                    for label in labels:
                        new_counts.append(bucket_to_count.get(label, 0))
                    counts = new_counts

                    # Use labels as bucket names
                    buckets = labels
                # else: for labeled_counter without predefined labels, use data labels as-is

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

        if crash_event_metrics is not None:
            for metric in self.config.get("crash_event_metrics", {}):
                df = crash_event_metrics[metric]
                if segment == "All":
                    # For 'All' segment, sum across all segments for this branch
                    crash_count = df[df["branch"] == branch]["crash_count"].sum()
                else:
                    # For specific segment, get the crash count for this branch and segment
                    subset = df[(df["segment"] == segment) & (df["branch"] == branch)]
                    crash_count = subset["crash_count"].sum() if len(subset) > 0 else 0

                results[branch][segment]["crash_event_metrics"][metric] = crash_count
                print(f"    segment={segment} crash event: {metric} = {crash_count}")

    def getResults(self):
        if self.config["is_experiment"] is True:
            return self.getResultsForExperiment()
        else:
            return self.getResultsForNonExperiment()

    def getResultsForNonExperiment(self):
        # Get data for each pageload event metric.
        event_metrics = {}
        for metric in self.config["pageload_event_metrics"]:
            event_metrics[metric] = self.getPageloadEventDataNonExperiment(metric)
            print(event_metrics[metric])

        # Get data for each crash event metric.
        crash_event_metrics = {}
        for metric in self.config.get("crash_event_metrics", {}):
            crash_event_metrics[metric] = self.getCrashEventDataNonExperiment(metric)
            print(f"Crash event data for {metric}: {crash_event_metrics[metric]}")

        # Get data for each histogram in this segment.
        histograms, remove = self._processHistogramsInParallel(
            self.config["histograms"], is_experiment=False
        )

        for hist in remove:
            if hist in self.config["histograms"]:
                del self.config["histograms"][hist]

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
                    crash_event_metrics,
                )

        results["queries"] = self.queries
        return results

    def getResultsForExperiment(self):
        # Get data for each pageload event metric.
        event_metrics = {}
        for metric in self.config["pageload_event_metrics"]:
            event_metrics[metric] = self.getPageloadEventData(metric)
            print(event_metrics[metric])

        # Get data for each crash event metric.
        crash_event_metrics = {}
        for metric in self.config.get("crash_event_metrics", {}):
            crash_event_metrics[metric] = self.getCrashEventData(metric)
            print(f"Crash event data for {metric}: {crash_event_metrics[metric]}")

        # Separate regular histograms from labeled_percentiles and quantity_percentiles metrics
        regular_histograms = {}
        labeled_percentiles = {}
        quantity_percentiles = {}
        for hist in self.config["histograms"]:
            kind = self.config["histograms"][hist].get("kind")
            if kind == "labeled_percentiles":
                labeled_percentiles[hist] = self.config["histograms"][hist]
            elif kind == "quantity_percentiles":
                quantity_percentiles[hist] = self.config["histograms"][hist]
            else:
                regular_histograms[hist] = self.config["histograms"][hist]

        # Get data for regular histograms
        histograms, remove = self._processHistogramsInParallel(
            regular_histograms, is_experiment=True
        )

        # Remove invalid histogram data.
        for hist in remove:
            if hist in self.config["histograms"]:
                print(f"Empty dataset found, removing: {hist}.")
                del self.config["histograms"][hist]

        # Get data for labeled_percentiles metrics
        labeled_percentiles_data = {}
        for hist in labeled_percentiles:
            print(f"Fetching labeled percentiles data for: {hist}")
            df = self.getLabeledPercentilesData(self.config, hist)
            if df is not None and not df.empty:
                labeled_percentiles_data[hist] = df

        # Get data for quantity_percentiles metrics
        quantity_percentiles_data = {}
        for hist in quantity_percentiles:
            print(f"Fetching quantity percentiles data for: {hist}")
            df = self.getQuantityPercentilesData(self.config, hist)
            if df is not None and not df.empty:
                quantity_percentiles_data[hist] = df

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
                    crash_event_metrics,
                )

                # Process labeled_percentiles data
                self.collectLabeledPercentilesResults(
                    results,
                    branch_name,
                    segment,
                    labeled_percentiles_data,
                )

                # Process quantity_percentiles data
                self.collectQuantityPercentilesResults(
                    results,
                    branch_name,
                    segment,
                    quantity_percentiles_data,
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
            "single_os_filter": self._get_single_os_filter(),
        }

        query = t.render(context)
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
            "single_os_filter": self._get_single_os_filter(),
        }
        query = t.render(context)
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
            "distribution_type": self.config["histograms"][histogram].get(
                "distribution_type"
            ),
            "single_os_filter": self._get_single_os_filter(),
        }
        query = t.render(context)
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def generateHistogramQuery_OS_segments_non_experiment_glean(self, histogram):
        t = get_template("other/glean/histogram_os_segments.sql")

        branches = self.config["branches"].copy()

        # Add safe_name field for SQL identifiers (replace spaces with underscores)
        for branch in branches:
            branch["safe_name"] = branch["name"].replace(" ", "_")
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

        # Determine if desktop and Android sections are needed based on configured segments
        desktop_segments = {"Windows", "Linux", "Mac"}
        android_segments = {"Android"}
        configured_segments = set(self.config.get("segments", []))

        needs_desktop = (
            bool(desktop_segments & configured_segments)
            and self.config["histograms"][histogram]["available_on_desktop"]
        )
        needs_android = (
            bool(android_segments & configured_segments)
            and self.config["histograms"][histogram]["available_on_android"]
        )

        # Create Android version of custom condition if it exists
        for branch in branches:
            if "custom_condition" in branch:
                branch["custom_condition_android"] = branch["custom_condition"].replace(
                    "m.metrics", "f.metrics"
                )

        # Render the prerequisite CTE template with date variables
        prerequisite_cte_template = self.config.get("prerequisite_ctes", "")
        if prerequisite_cte_template:
            # Use string replacement for simple template variables
            rendered_prerequisite_ctes = prerequisite_cte_template.replace(
                "{{start_date}}", branches[0]["startDate"]
            ).replace("{{end_date}}", branches[0]["endDate"])
        else:
            rendered_prerequisite_ctes = ""

        # Get sampling/limiting configuration
        sampling_config = self._calculateSamplePct(histogram)
        sample_threshold, sample_modulus = self._parseSamplingConfig(sampling_config)

        # For backward compatibility with templates, convert to old format if it's manual percentage
        if sample_modulus is None and sample_threshold is not None:
            sample_pct = sample_threshold
        else:
            sample_pct = sample_threshold

        # Determine if using shared dates or per-branch dates
        # Check if all branches have the same date range
        first_start = branches[0]["startDate"]
        first_end = branches[0]["endDate"]
        all_same_dates = all(
            b["startDate"] == first_start and b["endDate"] == first_end
            for b in branches
        )
        use_shared_dates = all_same_dates

        context = {
            "histogram": histogram,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
            "distribution_type": self.config["histograms"][histogram].get(
                "distribution_type"
            ),
            "needs_desktop": needs_desktop,
            "needs_android": needs_android,
            "branches": branches,
            "prerequisite_ctes": rendered_prerequisite_ctes,
            "use_shared_dates": use_shared_dates,
            "start_date": branches[0]["startDate"],
            "end_date": branches[0]["endDate"],
            "channel": branches[0]["channel"],
            "sample_pct": sample_pct,
            "sample_threshold": sample_threshold,
            "sample_modulus": sample_modulus or 100,  # Default to 100 for old templates
            "single_os_filter": self._get_single_os_filter(),
        }

        query = t.render(context)
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query

    def checkForExistingData(self, filename):
        if self.skipCache:
            df = None
        else:
            try:
                df = pd.read_pickle(filename)
                print(f"Found local data in {filename}")
            except Exception:
                df = None
        return df

    def getHistogramDataNonExperiment(self, config, histogram):
        slug = config["slug"]
        hist_name = histogram.split(".")[-1]
        filename = os.path.join(self.dataDir, f"{slug}-{hist_name}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        # Determine which query generator to use
        if config_has_custom_branches(self.config) or config_has_custom_segments(
            self.config
        ):
            query = self.generateHistogramQuery_OS_segments(
                histogram, use_custom_conditions=True
            )
        elif segments_are_all_OS(self.config["segments"]):
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

    def getHistogramData(self, config, histogram):
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

    def generateLabeledPercentilesQuery_OS_segments_glean(self, histogram):
        t = get_template("experiment/glean/labeled_percentiles_os_segments.sql")

        context = {
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
            "single_os_filter": self._get_single_os_filter(),
        }
        query = t.render(context)
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        self.queries.append({"name": f"Labeled Percentiles: {histogram}", "query": query})
        return query

    def getLabeledPercentilesData(self, config, histogram):
        slug = config["slug"]
        hist_name = histogram.split(".")[-1]
        filename = os.path.join(self.dataDir, f"{slug}-{hist_name}-percentiles.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generateLabeledPercentilesQuery_OS_segments_glean(histogram)
        else:
            print("No current support for generic non-experiment queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()

        print(f"Writing '{slug}' labeled percentiles results for {histogram} to disk.")
        df.to_pickle(filename)
        return df

    def collectLabeledPercentilesResults(
        self,
        results,
        branch,
        segment,
        labeled_percentiles_data,
    ):
        """Process labeled_percentiles data and store median, p75, p95 for each label."""
        for histogram, df in labeled_percentiles_data.items():
            # Filter data for this branch and segment
            if segment == "All":
                subset = df[df["branch"] == branch]
            else:
                subset = df[(df["segment"] == segment) & (df["branch"] == branch)]

            if subset.empty:
                continue

            # Extract labels and percentiles
            labels = list(subset["label"])
            medians = [float(x) for x in subset["median"]]
            p75s = [float(x) for x in subset["p75"]]
            p95s = [float(x) for x in subset["p95"]]
            counts = [int(x) for x in subset["sample_count"]]

            # Store in results
            hist_name = histogram.split(".")[-1]
            results[branch][segment]["histograms"][histogram] = {
                "labels": labels,
                "medians": medians,
                "p75s": p75s,
                "p95s": p95s,
                "counts": counts,
            }

            print(
                f"    segment={segment} labeled percentiles: {hist_name} = {len(labels)} labels"
            )

    def generateQuantityPercentilesQuery_OS_segments_glean(self, histogram):
        t = get_template("experiment/glean/quantity_percentiles_os_segments.sql")

        context = {
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
            "single_os_filter": self._get_single_os_filter(),
        }
        query = t.render(context)
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        self.queries.append({"name": f"Quantity Percentiles: {histogram}", "query": query})
        return query

    def getQuantityPercentilesData(self, config, histogram):
        slug = config["slug"]
        hist_name = histogram.split(".")[-1]
        filename = os.path.join(self.dataDir, f"{slug}-{hist_name}-quantity-percentiles.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generateQuantityPercentilesQuery_OS_segments_glean(histogram)
        else:
            print("No current support for generic non-experiment queries.")
            sys.exit(1)

        print("Running query:\n" + query)
        job = self.client.query(query)
        df = job.to_dataframe()

        print(f"Writing '{slug}' quantity percentiles results for {histogram} to disk.")
        df.to_pickle(filename)
        return df

    def collectQuantityPercentilesResults(
        self,
        results,
        branch,
        segment,
        quantity_percentiles_data,
    ):
        """Process quantity_percentiles data and store median, p75, p95, sum."""
        for histogram, df in quantity_percentiles_data.items():
            # Filter data for this branch and segment
            if segment == "All":
                subset = df[df["branch"] == branch]
            else:
                subset = df[(df["segment"] == segment) & (df["branch"] == branch)]

            if subset.empty:
                continue

            # Extract percentiles and sum
            if len(subset) > 0:
                row = subset.iloc[0]
                median = float(row["median"]) if row["median"] is not None else 0.0
                p75 = float(row["p75"]) if row["p75"] is not None else 0.0
                p95 = float(row["p95"]) if row["p95"] is not None else 0.0
                total_sum = float(row["total_sum"]) if row["total_sum"] is not None else 0.0
                count = int(row["sample_count"])

                # Store in results
                results[branch][segment]["histograms"][histogram] = {
                    "median": median,
                    "p75": p75,
                    "p95": p95,
                    "sum": total_sum,
                    "count": count,
                }

                hist_name = histogram.split(".")[-1]
                print(
                    f"    segment={segment} quantity percentiles: {hist_name} = median:{median:.0f}, p75:{p75:.0f}, p95:{p95:.0f}, n:{count}"
                )

    def getPageloadEventDataNonExperiment(self, metric):
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
        import time

        start_time = time.time()
        job = self.client.query(query)
        df = job.to_dataframe()
        query_time = time.time() - start_time
        print(f"Query completed in {query_time:.1f} seconds")
        print(f"Writing '{slug}' pageload event results to disk.")
        df.to_pickle(filename)
        return df

    def getPageloadEventData(self, metric):
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
        import time

        start_time = time.time()
        job = self.client.query(query)
        df = job.to_dataframe()
        query_time = time.time() - start_time
        print(f"Query completed in {query_time:.1f} seconds")
        print(f"Writing '{slug}' pageload event results to disk.")
        df.to_pickle(filename)
        return df

    def generateCrashEventQuery_OS_segments_non_experiment(self, metric):
        t = get_template("other/glean/crash_events_os_segments.sql")

        branches = self.config["branches"].copy()

        for branch in branches:
            branch["safe_name"] = branch["name"].replace(" ", "_")

            if "custom_condition" in branch:
                branch["custom_condition_android"] = branch["custom_condition"].replace(
                    "m.metrics", "f.metrics"
                )

        for i in range(len(branches)):
            if "version" in branches[i]:
                version = branches[i]["version"]
                branches[i][
                    "ver_condition"
                ] = f"AND SPLIT(client_info.app_display_version, '.')[offset(0)] = \"{version}\""
            if "architecture" in branches[i]:
                arch = branches[i]["architecture"]
                branches[i][
                    "arch_condition"
                ] = f'AND client_info.architecture = "{arch}"'

            branches[i]["last"] = i == len(branches) - 1

            # Setup glean_conditions from custom_condition
            branches[i]["glean_conditions"] = []
            if "custom_condition" in branches[i]:
                branches[i]["glean_conditions"].append(
                    f"AND ({branches[i]['custom_condition']})"
                )

        context = {
            "branches": branches,
            "single_os_filter": self._get_single_os_filter(),
        }

        query = t.render(context)
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
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
            "single_os_filter": self._get_single_os_filter(),
        }
        query = t.render(context)
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        self.queries.append({"name": f"Crash event: {metric}", "query": query})
        return query

    def getCrashEventDataNonExperiment(self, metric):
        slug = self.config["slug"]
        filename = os.path.join(self.dataDir, f"{slug}-crash-events-{metric}.pkl")

        df = self.checkForExistingData(filename)
        if df is not None:
            return df

        if segments_are_all_OS(self.config["segments"]):
            query = self.generateCrashEventQuery_OS_segments_non_experiment(metric)
        else:
            print("Generic non-experiment crash event query currently not supported.")
            sys.exit(1)

        print("Running query:\n" + query)
        import time

        start_time = time.time()
        job = self.client.query(query)
        df = job.to_dataframe()
        query_time = time.time() - start_time
        print(f"Query completed in {query_time:.1f} seconds")
        print(f"Writing '{slug}' crash event results to disk.")
        df.to_pickle(filename)
        return df

    def getCrashEventData(self, metric):
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
        import time

        start_time = time.time()
        job = self.client.query(query)
        df = job.to_dataframe()
        query_time = time.time() - start_time
        print(f"Query completed in {query_time:.1f} seconds")
        print(f"Writing '{slug}' crash event results to disk.")
        df.to_pickle(filename)
        return df

    def generateHistogramQuery_OS_segments(
        self, histogram: str, use_custom_conditions: bool = True
    ):
        """Generate histogram query for OS segments with flexible conditions."""
        t = get_template("other/glean/histogram_os_segments.sql")

        branches = self.config["branches"].copy()

        # Add safe_name field for SQL identifiers (replace spaces with underscores)
        # Also prepare Android-specific custom conditions
        for branch in branches:
            branch["safe_name"] = branch["name"].replace(" ", "_")

            # Create Android version of custom condition if it exists
            if "custom_condition" in branch:
                branch["custom_condition_android"] = branch["custom_condition"].replace(
                    "m.metrics", "f.metrics"
                )

        # Process branches for standard conditions if needed
        if not use_custom_conditions:
            for i in range(len(branches)):
                if "version" in branches[i]:
                    version = branches[i]["version"]
                    branches[i][
                        "ver_condition"
                    ] = f"AND SPLIT(client_info.app_display_version, '.')[offset(0)] = \"{version}\""
                if "architecture" in branches[i]:
                    arch = branches[i]["architecture"]
                    branches[i][
                        "arch_condition"
                    ] = f'AND client_info.architecture = "{arch}"'
                if "glean_conditions" in branches[i]:
                    branches[i]["glean_conditions"] = branches[i]["glean_conditions"]

        # Render the prerequisite CTE template with date variables
        prerequisite_cte_template = self.config.get("prerequisite_ctes", "")
        if prerequisite_cte_template:
            # Use string replacement for simple template variables
            rendered_prerequisite_ctes = prerequisite_cte_template.replace(
                "{{start_date}}", branches[0]["startDate"]
            ).replace("{{end_date}}", branches[0]["endDate"])
        else:
            rendered_prerequisite_ctes = ""

        # Determine if using shared dates or per-branch dates
        # Check if all branches have the same date range
        first_start = branches[0]["startDate"]
        first_end = branches[0]["endDate"]
        all_same_dates = all(
            b["startDate"] == first_start and b["endDate"] == first_end
            for b in branches
        )
        use_shared_dates = all_same_dates

        # Get sampling/limiting configuration
        sampling_config = self._calculateSamplePct(histogram)
        sample_threshold, sample_modulus = self._parseSamplingConfig(sampling_config)

        # For backward compatibility with templates, convert to old format if it's manual percentage
        if sample_modulus is None and sample_threshold is not None:
            sample_pct = sample_threshold
        else:
            sample_pct = sample_threshold

        # Determine if desktop and Android sections are needed based on configured segments
        desktop_segments = {"Windows", "Linux", "Mac"}
        android_segments = {"Android"}
        configured_segments = set(self.config.get("segments", []))

        needs_desktop = (
            bool(desktop_segments & configured_segments)
            and self.config["histograms"][histogram]["available_on_desktop"]
        )
        needs_android = (
            bool(android_segments & configured_segments)
            and self.config["histograms"][histogram]["available_on_android"]
        )

        context = {
            "histogram": histogram,
            "branches": branches,
            "prerequisite_ctes": rendered_prerequisite_ctes,
            "use_shared_dates": use_shared_dates,
            "start_date": branches[0]["startDate"] if use_shared_dates else None,
            "end_date": branches[0]["endDate"] if use_shared_dates else None,
            "channel": branches[0]["channel"] if use_shared_dates else None,
            "sample_pct": sample_pct,
            "sample_threshold": sample_threshold,
            "sample_modulus": sample_modulus or 100,
            "needs_desktop": needs_desktop,
            "needs_android": needs_android,
            "available_on_desktop": self.config["histograms"][histogram][
                "available_on_desktop"
            ],
            "available_on_android": self.config["histograms"][histogram][
                "available_on_android"
            ],
            "distribution_type": self.config["histograms"][histogram].get(
                "distribution_type"
            ),
            "single_os_filter": self._get_single_os_filter(),
        }

        query = t.render(context)
        # Remove empty lines before returning
        query = "".join([s for s in query.strip().splitlines(True) if s.strip()])
        # Clean up any trailing UNION ALL before the closing parenthesis
        import re

        query = re.sub(
            r"\s*UNION ALL\s*\)\s*s\s*GROUP BY",
            ") s\nGROUP BY",
            query,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        self.queries.append({"name": f"Histogram: {histogram}", "query": query})
        return query
