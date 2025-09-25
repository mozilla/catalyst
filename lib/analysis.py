from scipy import stats
import numpy as np


# Expand the histogram into an array of values
def flatten_histogram(bins, counts):
    array = []
    for i in range(len(bins)):
        for j in range(1, int(counts[i] / 2.0)):
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
    density = []
    cdf = []
    cum = 0
    for i in range(len(counts)):
        density.append(float(counts[i] / n))
        cum = cum + counts[i]
        cdf.append(float(cum))
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


def calculate_histogram_stats(bins, counts, data):
    # Calculate mean, std, and var
    [mean, var, std, n] = calc_histogram_mean_var(bins, counts)
    data["mean"] = mean
    data["std"] = std
    data["var"] = var
    data["n"] = n

    # Calculate densities
    [density, cdf] = calc_histogram_density(counts, n)
    data["pdf"]["cdf"] = cdf
    data["pdf"]["density"] = density
    data["pdf"]["values"] = bins

    # Calculate quantiles
    [quantiles, vals] = calc_histogram_quantiles(bins, density)
    data["quantiles"] = quantiles
    data["quantile_vals"] = vals


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


def createNumericalTemplate():
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
    }
    return template


def createCategoricalTemplate():
    template = {"desc": "", "labels": [], "counts": [], "ratios": [], "sum": 0}
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
                "scalar": [...]
            }
        """
        print("Processing unified data types...")

        # Process each data type with unified algorithms
        self.processNumericalMetrics(transformedData["numerical"])
        self.processCategoricalMetrics(transformedData["categorical"])
        self.processScalarMetrics(transformedData["scalar"])

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
            result_location["labels"] = labels
            result_location["counts"] = counts
            total = sum(counts)
            result_location["sum"] = total
            ratios = [x / total for x in counts]
            result_location["ratios"] = ratios

            # Calculate uplift for non-control branches
            if branch != self.control:
                # Find corresponding control data
                if metric_name in self.results[self.control][segment]["categorical"]:
                    control_result = self.results[self.control][segment]["categorical"][
                        metric_name
                    ]
                    if "ratios" in control_result:
                        control_ratios = control_result["ratios"]
                        uplift = []
                        for i in range(len(ratios)):
                            uplift.append((ratios[i] - control_ratios[i]) * 100)
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
            scalar_value = data.get("crash_count", data.get("count", 0))
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
