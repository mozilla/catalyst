from scipy import stats, sparse
import numpy as np
import pandas as pd
import hashlib
import pyarrow as pa
import pyarrow.compute as pc


# Expand the histogram into an array of values
def flatten_histogram(bins, counts):
    # Convert counts to int in case they're decimal.Decimal from BigQuery
    counts = [int(c) for c in counts]
    array = []
    for i in range(len(bins)):
        for j in range(1, counts[i] // 2):
            array.append(bins[i])
    return array


# effect size calculation for t-test
def calc_cohen_d(x1, x2, s1, s2, n1, n2):
    # Calculate pooled standard deviation
    pooled_variance = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_variance)

    # Handle edge cases where pooled standard deviation is very small
    if pooled_std < 1e-10:  # Essentially zero variance
        # If means are identical, effect size is 0; otherwise return a large but finite value
        if abs(x1 - x2) < 1e-10:
            return 0.0
        else:
            return np.sign(x1 - x2) * 10.0  # Large effect size, capped at 10

    effect_size = (x1 - x2) / pooled_std

    # Cap effect size to reasonable bounds to avoid overflow issues
    return np.clip(effect_size, -10.0, 10.0)


# effect size calculation for mwu
def rank_biserial_correlation(n1, n2, U):
    return 1 - 2 * U / (n1 * n2)


# Calculate two-way t-test with unequal sample size and unequal variances.
# Return the t-value, p-value, and effect size via cohen's d.
def calc_t_test(x1, x2, s1, s2, n1, n2):
    s_prime = np.sqrt((s1**2 / n1) + (s2**2) / n2)
    if s_prime == 0:
        raise ZeroDivisionError("t-test is undefined with zero variance")

    t_value = (x1 - x2) / s_prime

    df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    )
    p_value = 2 * (1 - (stats.t.cdf(abs(t_value), df)))
    effect_size = calc_cohen_d(x1, x2, s1, s2, n1, n2)
    return [t_value, p_value, effect_size]


def create_subsample(bins, counts, sample_size=100000):
    total_counts = sum(counts)
    if total_counts <= sample_size:
        return flatten_histogram(bins, counts)

    ratio = total_counts / sample_size
    subsample = []
    for i in range(len(bins)):
        subsample.extend(np.repeat(bins[i], counts[i] / ratio))
    return subsample


def calc_cdf_from_density(density, vals):
    cdf = []
    sum = 0
    for i in range(0, len(density) - 2):
        width = vals[i + 1] - vals[i]
        cdf_val = sum + density[i] * width
        sum = cdf_val
        cdf.append(cdf_val)

    width = vals[-1] - vals[-2]
    cdf_val = sum + density[-1] * width
    sum = cdf_val
    cdf.append(cdf_val)
    return cdf


# TODO: Interpolate the quantiles.
def calc_histogram_quantiles(bins, density):
    vals = []
    quantiles = []
    q = 0
    for i in range(len(bins)):
        q = q + density[i]
        vals.append(bins[i])
        quantiles.append(q)

    return [quantiles, vals]


def calc_histogram_density(counts, n):
    # Convert counts to float in case they're decimal.Decimal from BigQuery
    counts = [float(c) for c in counts]
    n = float(n)
    density = []
    cdf = []
    cum = 0
    for i in range(len(counts)):
        density.append(counts[i] / n)
        cum = cum + counts[i]
        cdf.append(cum)
    cdf = [x / cum for x in cdf]
    return [density, cdf]


def calc_histogram_mean_var(bins, counts):
    mean = 0
    n = 0
    for i in range(len(bins)):
        bucket = float(bins[i])
        count = float(counts[i])
        n = n + count
        mean = mean + bucket * count
    mean = float(mean) / float(n)

    var = 0
    for i in range(len(bins)):
        bucket = float(bins[i])
        count = float(counts[i])
        var = var + count * (bucket - mean) ** 2
    var = float(var) / float(n)
    std = np.sqrt(var)

    return [mean, var, std, n]


def calc_histogram_percentile(bins, counts, q):
    """Calculate the q-th percentile (0 < q < 1) from histogram bins and counts."""
    # float, not int: bootstrap replicate counts are fractional weighted sums,
    # and BigQuery may return decimal.Decimal
    counts = [float(c) for c in counts]
    total_count = sum(counts)
    if total_count == 0:
        return 0.0

    # Find the bin containing the q-th percentile
    target = total_count * q
    cumulative = 0

    for i, count in enumerate(counts):
        cumulative += count
        if cumulative >= target:
            # Linear interpolation within the bin
            if count == 0:
                return bins[i]

            # How far into the bin is the target?
            remaining = target - (cumulative - count)
            fraction = remaining / count

            # If we're at the last bin, just return the bin value
            if i == len(bins) - 1:
                return bins[i]

            # Interpolate between this bin and the next
            bin_width = bins[i + 1] - bins[i] if i < len(bins) - 1 else 0
            return bins[i] + fraction * bin_width

    # Fallback - should not reach here
    return bins[-1] if bins else 0.0


def calculate_histogram_stats(bins, counts, data):
    # Calculate mean, std, and var
    [mean, var, std, n] = calc_histogram_mean_var(bins, counts)
    data["mean"] = mean
    data["std"] = std
    data["var"] = var
    data["n"] = n

    # Calculate median, p75, p95
    data["median"] = calc_histogram_percentile(bins, counts, 0.5)
    data["p75"] = calc_histogram_percentile(bins, counts, 0.75)
    data["p95"] = calc_histogram_percentile(bins, counts, 0.95)

    # Calculate densities
    [density, cdf] = calc_histogram_density(counts, n)
    data["pdf"]["cdf"] = cdf
    data["pdf"]["density"] = density
    data["pdf"]["values"] = bins

    # Calculate quantiles
    [quantiles, vals] = calc_histogram_quantiles(bins, density)
    data["quantiles"] = quantiles
    data["quantile_vals"] = vals

    # Check for overflow bucket issues
    # If >90% of data is in the last bucket, the histogram is likely saturated
    if len(counts) > 0 and n > 0:
        last_bucket_ratio = float(counts[-1]) / float(n)
        if last_bucket_ratio > 0.90:
            data["overflow_warning"] = {
                "last_bucket_ratio": last_bucket_ratio,
                "last_bucket_value": bins[-1],
                "message": f"WARNING: {last_bucket_ratio*100:.1f}% of data is in the overflow bucket at {bins[-1]}. "
                f"This histogram's maximum bucket is too small for the measured values, "
                f"making statistical analysis unreliable.",
            }


def calculate_histogram_tests_subsampling(control_data, branch_data, result):
    bins_control = control_data["bins"]
    counts_control = control_data["counts"]
    control_sample = create_subsample(bins_control, counts_control)

    bins_branch = branch_data["bins"]
    counts_branch = branch_data["counts"]
    branch_sample = create_subsample(bins_branch, counts_branch)

    # Calculate t-test and effect
    x1 = np.mean(control_sample)
    s1 = np.std(control_sample)
    n1 = len(control_sample)
    x2 = np.mean(branch_sample)
    s2 = np.std(branch_sample)
    n2 = len(branch_sample)
    effect = calc_cohen_d(x1, x2, s1, s2, n1, n2)
    [t, p] = stats.ttest_ind(control_sample, branch_sample)
    result["tests"]["ttest"] = {}
    result["tests"]["ttest"]["score"] = t
    result["tests"]["ttest"]["p-value"] = p
    result["tests"]["ttest"]["effect"] = effect

    # Calculate mwu-test
    [U, p] = stats.mannwhitneyu(control_sample, branch_sample)
    r = rank_biserial_correlation(n1, n2, U)
    result["tests"]["mwu"] = {}
    result["tests"]["mwu"]["score"] = U
    result["tests"]["mwu"]["p-value"] = p
    result["tests"]["mwu"]["effect"] = r

    # Calculate ks-test
    [D, p] = stats.ks_2samp(control_sample, branch_sample)
    result["tests"]["ks"] = {}
    result["tests"]["ks"]["score"] = D
    result["tests"]["ks"]["p-value"] = p
    result["tests"]["ks"]["effect"] = D


def calculate_histogram_ttest(bins, counts, data, control):
    mean_control = control["mean"]
    std_control = control["std"]
    n_control = control["n"]

    mean = data["mean"]
    std = data["std"]
    n = data["n"]

    # Calculate t-test
    [t_value, p_value, effect] = calc_t_test(
        mean, mean_control, std, std_control, n, n_control
    )
    data["tests"]["ttest"] = {}
    data["tests"]["ttest"]["score"] = t_value
    data["tests"]["ttest"]["p-value"] = p_value
    data["tests"]["ttest"]["effect"] = effect


def calc_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return [m, se, m - h, m + h]


BOOTSTRAP_COLUMNS = ["segment", "branch", "bucket", "counts", "bootstrap_replicates"]

# Cap on weight-matrix cells per chunk (~256 MB of float64).
BOOTSTRAP_WEIGHT_CELLS = 32_000_000


def _bootstrap_seed(slug: str, metric_name: str, branch_name: str, segment: str) -> int:
    """Deterministic seed so CI bounds don't jitter between runs. Per-segment,
    so segments summed into "All" draw independent weights."""
    key = f"{slug}:{metric_name}:{branch_name}:{segment}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


# Rows buffered from the query stream and bootstrapped together: enough to
# amortize per-batch overhead, few enough that the buffer stays a few hundred
# MB however many clients the experiment has.
BOOTSTRAP_BATCH_ROWS = 250_000

_SPLITMIX_GAMMA = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(x):
    """splitmix64 finalizer, in place over a uint64 array."""
    x += _SPLITMIX_GAMMA
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _bootstrap_weights(unit_hashes, iterations, seed):
    """[units x iterations] Exponential(1) weights, each a hash of (unit id,
    iteration, seed). A unit's weight doesn't depend on which other units are
    present or their order, so a stream can be bootstrapped batch by batch."""
    keys = _splitmix64(
        (np.uint64(seed) << np.uint64(32)) | iterations.astype(np.uint64)
    )
    bits = _splitmix64(np.bitwise_xor.outer(unit_hashes, keys))
    bits >>= np.uint64(11)
    weights = bits.astype(np.float64)
    del bits
    weights += 0.5
    weights *= 2.0**-53
    np.log(weights, out=weights)
    return np.negative(weights, out=weights)


class _GroupAccumulator:
    """Running counts and replicates for one (metric, segment, branch)."""

    def __init__(self, n_bootstrap, seed):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.vocab = None
        self.counts = np.zeros(0)
        self.replicates = np.zeros((n_bootstrap, 0))

    def _extend_vocab(self, buckets):
        seen = pc.unique(buckets)
        if self.vocab is None:
            new = seen
            self.vocab = seen
        else:
            new = seen.filter(pc.invert(pc.is_in(seen, value_set=self.vocab)))
            self.vocab = pa.concat_arrays([self.vocab, new.cast(self.vocab.type)])
        if len(new):
            self.counts = np.concatenate([self.counts, np.zeros(len(new))])
            self.replicates = np.hstack(
                [self.replicates, np.zeros((self.n_bootstrap, len(new)))]
            )

    def add(self, units, bucket_counts):
        n_units = len(bucket_counts)
        if n_units == 0:
            return
        # CSR straight from the list offsets: the SQL emits one entry per
        # (unit, bucket), so rows need no sorting or deduplication.
        entries = bucket_counts.flatten()
        buckets = pc.struct_field(entries, "bucket")
        self._extend_vocab(buckets)
        offsets = bucket_counts.offsets.to_numpy()
        matrix = sparse.csr_matrix(
            (
                pc.struct_field(entries, "counts")
                .to_numpy(zero_copy_only=False)
                .astype(np.float64),
                pc.index_in(buckets, value_set=self.vocab).to_numpy(),
                offsets - offsets[0],
            ),
            shape=(n_units, len(self.vocab)),
        )
        self.counts += np.asarray(matrix.sum(axis=0)).ravel()

        unit_hashes = pd.util.hash_array(units.to_numpy(zero_copy_only=False))
        # Weights are drawn a chunk of iterations at a time: the full
        # [units x iterations] matrix is 8 bytes per unit per iteration.
        chunk = max(1, min(self.n_bootstrap, BOOTSTRAP_WEIGHT_CELLS // n_units))
        transposed = matrix.T
        for start in range(0, self.n_bootstrap, chunk):
            iterations = np.arange(start, min(start + chunk, self.n_bootstrap))
            weights = _bootstrap_weights(unit_hashes, iterations, self.seed)
            self.replicates[iterations] += (transposed @ weights).T

    def frame(self, segment, branch):
        buckets = self.vocab.to_numpy(zero_copy_only=False)
        order = np.argsort(buckets, kind="stable")
        return pd.DataFrame(
            {
                "segment": segment,
                "branch": branch,
                "bucket": buckets[order],
                "counts": self.counts[order],
                "bootstrap_replicates": list(
                    np.ascontiguousarray(self.replicates[:, order].T)
                ),
            },
            columns=BOOTSTRAP_COLUMNS,
        )


def _buffered_tables(data):
    """Yield pyarrow Tables of about BOOTSTRAP_BATCH_ROWS rows from a Table,
    DataFrame, or iterable of RecordBatches/Tables."""
    if isinstance(data, pd.DataFrame):
        data = pa.Table.from_pandas(data, preserve_index=False)
    if isinstance(data, (pa.Table, pa.RecordBatch)):
        data = [data]
    buffered, rows = [], 0
    for item in data:
        for batch in item.to_batches() if isinstance(item, pa.Table) else [item]:
            buffered.append(batch)
            rows += batch.num_rows
            if rows >= BOOTSTRAP_BATCH_ROWS:
                yield pa.Table.from_batches(buffered)
                buffered, rows = [], 0
    if buffered:
        yield pa.Table.from_batches(buffered)


def bootstrap_stream(data, n_bootstrap, slug, metric_names, unit_col="client_id"):
    """Cluster (Bayesian) bootstrap replicates, resampling whole units rather
    than individual observations, for each metric in `metric_names`.
    `unit_col` is `client_id` for metrics-ping histograms and `document_id`
    for pageload-ping events, which carry no client_id.

    `data` is a BigQuery Arrow stream (or one Table/DataFrame) with one row
    per (unit, segment, branch), a `bucket_counts` list of {"bucket",
    "counts"} entries, and a `metric_name` column when it holds several
    metrics. Without that column every row belongs to the single metric in
    `metric_names`.

    Each unit gets an Exponential(1) weight per iteration, hashed from its id,
    so rows are consumed in bounded batches and the result is the same, up
    to floating-point summation order, for any row order or batch split. Memory is set by the batch size and the
    number of distinct buckets, not by the number of units.

    Returns {metric: DataFrame of (segment, branch, bucket, counts,
    bootstrap_replicates)}; a metric with no rows gets an empty frame.
    """
    groups = {}
    for table in _buffered_tables(data):
        has_metric = "metric_name" in table.column_names
        key_cols = (
            ["metric_name", "segment", "branch"]
            if has_metric
            else ["segment", "branch"]
        )
        for key in table.select(key_cols).group_by(key_cols).aggregate([]).to_pylist():
            metric = key["metric_name"] if has_metric else metric_names[0]
            if metric not in metric_names:
                continue
            mask = pc.and_(
                pc.equal(table["segment"], key["segment"]),
                pc.equal(table["branch"], key["branch"]),
            )
            if has_metric:
                mask = pc.and_(mask, pc.equal(table["metric_name"], metric))
            rows_idx = pc.indices_nonzero(mask)
            group_key = (metric, key["segment"], key["branch"])
            if group_key not in groups:
                groups[group_key] = _GroupAccumulator(
                    n_bootstrap,
                    _bootstrap_seed(slug, metric, key["branch"], key["segment"]),
                )
            groups[group_key].add(
                table[unit_col].take(rows_idx),
                table["bucket_counts"].take(rows_idx).combine_chunks(),
            )

    result = {}
    for metric in metric_names:
        frames = [
            groups[key].frame(key[1], key[2])
            for key in sorted(k for k in groups if k[0] == metric)
        ]
        # Columns stay present when there are no rows, so a metric with no
        # data reads as empty to callers instead of raising KeyError.
        result[metric] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
        )
    return result


def compute_client_cluster_bootstrap(
    client_df, n_bootstrap, slug, metric_name, unit_col="client_id"
):
    """bootstrap_stream for a single metric."""
    return bootstrap_stream(client_df, n_bootstrap, slug, [metric_name], unit_col)[
        metric_name
    ]


def _replicate_histograms(replicates):
    """Per-bucket replicate arrays -> one histogram per bootstrap iteration."""
    return np.array(replicates).T


def calc_bootstrap_ci(
    control_bins,
    control_replicates,
    treatment_bins,
    treatment_replicates,
    quantiles=(0.5, 0.75, 0.95),
    confidence=0.95,
):
    """CI on the relative lift (treatment / control - 1) at each quantile, as
    {quantile: {"lower", "upper"}}.

    Bounds are None unless both branches have cluster bootstrap replicates.
    Resampling pooled observations instead ignores clustering and gives
    miscalibrated intervals."""
    if not control_replicates or not treatment_replicates:
        return {q: {"lower": None, "upper": None} for q in quantiles}

    lifts = {q: [] for q in quantiles}
    for ctrl, trt in zip(
        _replicate_histograms(control_replicates),
        _replicate_histograms(treatment_replicates),
    ):
        for q in quantiles:
            c = calc_histogram_percentile(control_bins, ctrl, q)
            t = calc_histogram_percentile(treatment_bins, trt, q)
            lifts[q].append(t / c - 1 if c > 0 else 0.0)

    alpha = (1 - confidence) / 2
    return {
        q: {
            "lower": float(np.percentile(lifts[q], alpha * 100)),
            "upper": float(np.percentile(lifts[q], (1 - alpha) * 100)),
        }
        for q in quantiles
    }


def calc_cdf_band(bins, x_values, replicates, confidence=0.95):
    """Bootstrap confidence envelope for a CDF at `x_values`, or None bounds
    without cluster replicates (see calc_bootstrap_ci)."""
    if not replicates:
        return {"lower": [None] * len(x_values), "upper": [None] * len(x_values)}

    bins_arr = np.array([float(b) for b in bins])
    boot_cdfs = []
    for resampled in _replicate_histograms(replicates):
        total = resampled.sum()
        if total > 0:
            boot_cdfs.append(
                np.interp(x_values, bins_arr, np.cumsum(resampled) / total)
            )

    alpha = (1 - confidence) / 2
    return {
        "lower": np.percentile(boot_cdfs, alpha * 100, axis=0).tolist(),
        "upper": np.percentile(boot_cdfs, (1 - alpha) * 100, axis=0).tolist(),
    }


def createNumericalTemplate():
    ci_empty = {"lower": None, "upper": None}
    template = {
        "desc": "",
        "mean": 0,
        "confidence": {"min": 0, "max": 0},
        "se": 0,
        "var": 0,
        "std": 0,
        "n": 0,
        "pdf": {"values": [], "density": [], "cdf": []},
        "quantiles": [],
        "quantile_vals": [],
        "tests": {},
        "median_lift_ci": dict(ci_empty),
        "p75_lift_ci": dict(ci_empty),
        "p95_lift_ci": dict(ci_empty),
    }
    return template


def createCategoricalTemplate():
    template = {"desc": "", "labels": [], "counts": [], "sample_counts": [], "sum": 0}
    return template


def createScalarTemplate():
    template = {"desc": "", "count": 0, "n": 1}
    return template


def createResultsTemplate(config):
    template = {}
    for branch in config["branches"]:
        template[branch] = {}
        for segment in config["segments"]:
            template[branch][segment] = {
                "numerical": {},
                "categorical": {},
                "scalar": {},
                "labeled_percentiles": {},
                "quantity_percentiles": {},
            }
    return template


class DataAnalyzer:
    def __init__(self, config):
        self.config = config
        self.event_controldf = None
        self.control = self.config["branches"][0]
        self.results = createResultsTemplate(config)

    def processTelemetryData(self, transformedData):
        """Process transformed telemetry data organized by data types.

        Args:
            transformedData: Data organized by types: {
                "numerical": [...],
                "categorical": [...],
                "scalar": [...],
                "labeled_percentiles": [...]
            }
        """
        print("Processing unified data types...")

        # Process each data type with unified algorithms
        self.processNumericalMetrics(transformedData["numerical"])
        self.processCategoricalMetrics(transformedData["categorical"])
        self.processScalarMetrics(transformedData["scalar"])
        self.processLabeledPercentilesMetrics(
            transformedData.get("labeled_percentiles", [])
        )
        self.processQuantityPercentilesMetrics(
            transformedData.get("quantity_percentiles", [])
        )

        return self.results

    def processNumericalMetrics(self, numerical_metrics):
        """Process all numerical metrics with unified algorithms."""
        print(f"Processing {len(numerical_metrics)} numerical metrics")

        for metric_data in numerical_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["numerical"]:
                self.results[branch][segment]["numerical"][
                    metric_name
                ] = createNumericalTemplate()
            result_location = self.results[branch][segment]["numerical"][metric_name]
            result_location["desc"] = config.get("desc", f"{metric_name}")

            # Apply unified numerical analysis (works for both histograms and pageload events)
            bins = data["bins"]
            counts = data["counts"]
            calculate_histogram_stats(bins, counts, result_location)
            if "bootstrap_replicates" in data:
                result_location["bins"] = bins
                result_location["bootstrap_replicates"] = data["bootstrap_replicates"]

            # Print overflow warning if detected
            if "overflow_warning" in result_location:
                warning = result_location["overflow_warning"]
                print(
                    f"  ⚠️  {branch}/{segment}/{metric_name}: {warning['last_bucket_ratio']*100:.1f}% "
                    f"of data in overflow bucket (value={warning['last_bucket_value']})"
                )

            # Calculate statistical tests for non-control branches
            if branch != self.control:
                # Find corresponding control data
                control_data = None
                for control_metric in numerical_metrics:
                    if (
                        control_metric["branch"] == self.control
                        and control_metric["segment"] == segment
                        and control_metric["metric_name"] == metric_name
                        and control_metric["source_section"] == source_section
                    ):
                        control_data = control_metric["data"]
                        break

                if control_data:
                    calculate_histogram_tests_subsampling(
                        control_data, data, result_location
                    )

                    ci = calc_bootstrap_ci(
                        control_data["bins"],
                        control_data.get("bootstrap_replicates"),
                        data["bins"],
                        data.get("bootstrap_replicates"),
                    )
                    for name, q in (("median", 0.5), ("p75", 0.75), ("p95", 0.95)):
                        result_location[f"{name}_lift_ci"] = ci[q]
                    if data.get("bootstrap_replicates"):
                        result_location["bootstrap_iterations"] = len(
                            data["bootstrap_replicates"][0]
                        )

    def processCategoricalMetrics(self, categorical_metrics):
        """Process all categorical metrics with unified algorithms."""
        print(f"Processing {len(categorical_metrics)} categorical metrics")

        for metric_data in categorical_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["categorical"]:
                self.results[branch][segment]["categorical"][
                    metric_name
                ] = createCategoricalTemplate()
            result_location = self.results[branch][segment]["categorical"][metric_name]
            result_location["desc"] = config.get("desc", f"{metric_name}")

            # Apply unified categorical analysis
            labels = data["bins"]
            counts = data["counts"]
            sample_counts = data.get("sample_counts", [])
            result_location["labels"] = labels
            result_location["counts"] = counts
            result_location["sample_counts"] = sample_counts
            total = sum(counts)
            result_location["sum"] = total

            # Calculate uplift for non-control branches (percentage change in raw counts)
            if branch != self.control:
                if metric_name in self.results[self.control][segment]["categorical"]:
                    control_result = self.results[self.control][segment]["categorical"][
                        metric_name
                    ]
                    if "counts" in control_result:
                        control_counts = control_result["counts"]
                        uplift = []
                        for i in range(len(counts)):
                            if control_counts[i] > 0:
                                uplift.append(
                                    (
                                        (counts[i] - control_counts[i])
                                        / control_counts[i]
                                    )
                                    * 100
                                )
                            else:
                                uplift.append(0)
                        result_location["uplift"] = uplift

    def processScalarMetrics(self, scalar_metrics):
        """Process all scalar metrics with unified algorithms."""
        print(f"Processing {len(scalar_metrics)} scalar metrics")

        for metric_data in scalar_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["scalar"]:
                self.results[branch][segment]["scalar"][
                    metric_name
                ] = createScalarTemplate()
            result_location = self.results[branch][segment]["scalar"][metric_name]
            result_location["desc"] = config.get("desc", f"Total {metric_name}")

            # Apply unified scalar analysis
            # Handle both dictionary format and raw scalar values
            if isinstance(data, dict):
                scalar_value = data.get("crash_count", data.get("count", 0))
            else:
                # For crash events, data is stored as a raw integer
                scalar_value = data
            result_location["count"] = scalar_value
            result_location["n"] = 1

            # Calculate comparative statistics for non-control branches
            if branch != self.control:
                # Find corresponding control data
                if metric_name in self.results[self.control][segment]["scalar"]:
                    control_result = self.results[self.control][segment]["scalar"][
                        metric_name
                    ]
                    if "count" in control_result:
                        control_value = control_result["count"]
                        result_location["control_count"] = control_value
                        result_location["treatment_count"] = scalar_value

                        # Calculate relative change
                        if control_value > 0:
                            relative_change = (
                                (scalar_value - control_value) / control_value
                            ) * 100
                        else:
                            relative_change = scalar_value * 100

                        result_location["relative_change"] = relative_change

                        # Calculate absolute difference
                        absolute_diff = scalar_value - control_value
                        result_location["absolute_difference"] = absolute_diff

    def processLabeledPercentilesMetrics(self, labeled_percentiles_metrics):
        """Process all labeled_percentiles metrics - median, p75, p95 per label."""
        print(
            f"Processing {len(labeled_percentiles_metrics)} labeled_percentiles metrics"
        )

        for metric_data in labeled_percentiles_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure by data type
            if metric_name not in self.results[branch][segment]["labeled_percentiles"]:
                self.results[branch][segment]["labeled_percentiles"][metric_name] = {
                    "desc": config.get("desc", f"{metric_name}"),
                    "labels": [],
                    "medians": [],
                    "p75s": [],
                    "p95s": [],
                    "counts": [],
                }
            result_location = self.results[branch][segment]["labeled_percentiles"][
                metric_name
            ]

            # Store the data
            result_location["labels"] = data["labels"]
            result_location["medians"] = data["medians"]
            result_location["p75s"] = data["p75s"]
            result_location["p95s"] = data["p95s"]
            result_location["counts"] = data["counts"]

            # Calculate uplift for non-control branches
            if branch != self.control:
                if (
                    metric_name
                    in self.results[self.control][segment]["labeled_percentiles"]
                ):
                    control_result = self.results[self.control][segment][
                        "labeled_percentiles"
                    ][metric_name]
                    control_labels = control_result["labels"]
                    control_medians = control_result["medians"]
                    control_p75s = control_result["p75s"]
                    control_p95s = control_result["p95s"]

                    # Create label->percentile mappings for control
                    control_median_map = dict(zip(control_labels, control_medians))
                    control_p75_map = dict(zip(control_labels, control_p75s))
                    control_p95_map = dict(zip(control_labels, control_p95s))

                    # Calculate uplift for each label and percentile
                    median_uplifts = []
                    p75_uplifts = []
                    p95_uplifts = []

                    for i, label in enumerate(data["labels"]):
                        # Median uplift
                        if (
                            label in control_median_map
                            and control_median_map[label] > 0
                        ):
                            pct_change = (
                                (data["medians"][i] - control_median_map[label])
                                / control_median_map[label]
                            ) * 100
                            median_uplifts.append(pct_change)
                        else:
                            median_uplifts.append(0.0)

                        # p75 uplift
                        if label in control_p75_map and control_p75_map[label] > 0:
                            pct_change = (
                                (data["p75s"][i] - control_p75_map[label])
                                / control_p75_map[label]
                            ) * 100
                            p75_uplifts.append(pct_change)
                        else:
                            p75_uplifts.append(0.0)

                        # p95 uplift
                        if label in control_p95_map and control_p95_map[label] > 0:
                            pct_change = (
                                (data["p95s"][i] - control_p95_map[label])
                                / control_p95_map[label]
                            ) * 100
                            p95_uplifts.append(pct_change)
                        else:
                            p95_uplifts.append(0.0)

                    result_location["median_uplifts"] = median_uplifts
                    result_location["p75_uplifts"] = p75_uplifts
                    result_location["p95_uplifts"] = p95_uplifts

    def processQuantityPercentilesMetrics(self, quantity_percentiles_metrics):
        """Process all quantity_percentiles metrics - median, p75, p95 for scalar quantities."""
        print(
            f"Processing {len(quantity_percentiles_metrics)} quantity_percentiles metrics"
        )

        for metric_data in quantity_percentiles_metrics:
            branch = metric_data["branch"]
            segment = metric_data["segment"]
            metric_name = metric_data["metric_name"]
            source_section = metric_data["source_section"]
            config = metric_data["config"]
            data = metric_data["data"]

            print(f"  {branch}/{segment}: {source_section}.{metric_name}")

            # Store in results structure
            if metric_name not in self.results[branch][segment]["quantity_percentiles"]:
                self.results[branch][segment]["quantity_percentiles"][metric_name] = {
                    "desc": config.get("desc", f"{metric_name}"),
                    "median": 0,
                    "p75": 0,
                    "p95": 0,
                    "sum": 0,
                    "count": 0,
                }
            result_location = self.results[branch][segment]["quantity_percentiles"][
                metric_name
            ]

            # Store the data
            result_location["median"] = data["median"]
            result_location["p75"] = data["p75"]
            result_location["p95"] = data["p95"]
            result_location["sum"] = data["sum"]
            result_location["count"] = data["count"]

            # Calculate uplift for non-control branches
            if branch != self.control:
                if (
                    metric_name
                    in self.results[self.control][segment]["quantity_percentiles"]
                ):
                    control_result = self.results[self.control][segment][
                        "quantity_percentiles"
                    ][metric_name]
                    control_median = control_result["median"]
                    control_p75 = control_result["p75"]
                    control_p95 = control_result["p95"]
                    control_sum = control_result["sum"]

                    # Calculate uplifts for each percentile
                    if control_median > 0:
                        result_location["median_uplift"] = (
                            (data["median"] - control_median) / control_median
                        ) * 100
                    if control_p75 > 0:
                        result_location["p75_uplift"] = (
                            (data["p75"] - control_p75) / control_p75
                        ) * 100
                    if control_p95 > 0:
                        result_location["p95_uplift"] = (
                            (data["p95"] - control_p95) / control_p95
                        ) * 100
                    if control_sum > 0:
                        result_location["sum_uplift"] = (
                            (data["sum"] - control_sum) / control_sum
                        ) * 100
