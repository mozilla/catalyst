import json
import numpy as np
from scipy import interpolate
from django.template.loader import get_template
from airium import Airium
from bs4 import BeautifulSoup as bs


# These values are mostly hand-wavy that seem to
# fit the telemetry result impacts.
def get_cohen_effect_meaning(d):
    d_abs = abs(d)
    if d_abs <= 0.05:
        return "Small"
    if d_abs <= 0.1:
        return "Medium"
    else:
        return "Large"


def get_rank_biserial_corr_meaning(r):
    r_abs = abs(r)
    if r_abs <= 0.05:
        return "Small"
    if r_abs <= 0.1:
        return "Medium"
    else:
        return "Large"


# CubicSpline requires a monotonically increasing x.
# Remove duplicates.
def cubic_spline_prep(x, y):
    new_x = []
    new_y = []
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 0:
            new_x.append(x[i])
            new_y.append(y[i])
    return [new_x, new_y]


def cubic_spline_smooth(x, y, x_new):
    [x_prep, y_prep] = cubic_spline_prep(x, y)
    if len(x_prep) < 4:
        return [0.0] * len(x_new)
    tck = interpolate.splrep(x_prep, y_prep, k=3)
    y_new = interpolate.splev(x_new, tck, der=0)
    return y_new.tolist()


def find_value_at_quantile(values, cdf, q=0.95):
    if not values or not cdf:
        return None
    for i, e in reversed(list(enumerate(cdf))):
        if cdf[i] <= q:
            if i == len(cdf) - 1:
                return values[i]
            else:
                return values[i + 1]
    return None


def getIconForSegment(segment):
    iconMap = {
        "All": "fa-solid fa-globe",
        "Windows": "fa-brands fa-windows",
        "Linux": "fa-brands fa-linux",
        "Mac": "fa-brands fa-apple",
        "Android": "fa-brands fa-android",
    }
    if segment in iconMap:
        return iconMap[segment]
    else:
        return "fa-solid fa-chart-simple"


def flip_row_background(color):
    if color == "white":
        return "#ececec"
    else:
        return "white"


class ReportGenerator:
    def __init__(self, data):
        self.data = data
        self.doc = Airium()

    def get_metric_unit(self, metric):
        """Get the unit label for a metric based on its configuration."""
        if "input" in self.data and "histograms" in self.data["input"]:
            hist_config = self.data["input"]["histograms"].get(metric, {})

            # Check if it's a memory distribution
            if hist_config.get("distribution_type") == "memory_distribution":
                return "bytes"

            # Check if it's a timing distribution
            elif hist_config.get("distribution_type") == "timing_distribution":
                return "ms"

        return ""  # No unit label

    def createHeader(self):
        t = get_template("header.html")
        context = {"title": f"{self.data['slug']} experimental results"}
        self.doc(t.render(context))

    def endDocument(self):
        self.doc("</body>")
        return

    def createSidebar(self):
        t = get_template("sidebar.html")

        segments = []
        for segment in self.data["segments"]:
            entry = {
                "name": segment,
                "icon": getIconForSegment(segment),
                "numerical_metrics": [],
                "categorical_metrics": [],
                "scalar_metrics": [],
                "labeled_percentiles_metrics": [],
                "quantity_percentiles_metrics": [],
            }

            # Collect metrics from all branches for this segment
            all_metrics = {"numerical": set(), "categorical": set(), "scalar": set(), "labeled_percentiles": set(), "quantity_percentiles": set()}
            for branch in self.data["branches"]:
                branch_name = branch["name"] if isinstance(branch, dict) else branch
                if branch_name in self.data and segment in self.data[branch_name]:
                    for data_type in ["numerical", "categorical", "scalar", "labeled_percentiles", "quantity_percentiles"]:
                        if data_type in self.data[branch_name][segment]:
                            all_metrics[data_type].update(
                                self.data[branch_name][segment][data_type].keys()
                            )

            entry["numerical_metrics"] = sorted(list(all_metrics["numerical"]))
            entry["categorical_metrics"] = sorted(list(all_metrics["categorical"]))
            entry["scalar_metrics"] = sorted(list(all_metrics["scalar"]))
            entry["labeled_percentiles_metrics"] = sorted(list(all_metrics["labeled_percentiles"]))
            entry["quantity_percentiles_metrics"] = sorted(list(all_metrics["quantity_percentiles"]))

            segments.append(entry)

        ctx = {"segments": segments}
        self.doc(t.render(ctx))

    def createSummarySection(self):
        t = get_template("summary.html")
        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        row_background = "white"

        segments = []
        for segment in self.data["segments"]:
            numerical_metrics = []
            categorical_metrics = []

            # Process all data types
            for data_type in ["numerical", "categorical", "scalar", "labeled_percentiles", "quantity_percentiles"]:
                # Get all metrics for this data type across all branches
                all_metrics = set()
                for branch in self.data["branches"]:
                    branch_name = branch["name"] if isinstance(branch, dict) else branch
                    if (
                        branch_name in self.data
                        and segment in self.data[branch_name]
                        and data_type in self.data[branch_name][segment]
                    ):
                        all_metrics.update(
                            self.data[branch_name][segment][data_type].keys()
                        )

                for metric in sorted(all_metrics):
                    # Alternate between white and #ececec for row background
                    row_background = flip_row_background(row_background)
                    metric_name = metric

                    # Generate summary for categorical metrics
                    if data_type == "categorical":
                        branches = []
                        for branch in self.data["branches"]:
                            branch_name = (
                                branch["name"] if isinstance(branch, dict) else branch
                            )
                            if (
                                branch_name in self.data
                                and segment in self.data[branch_name]
                                and data_type in self.data[branch_name][segment]
                                and metric in self.data[branch_name][segment][data_type]
                                and "uplift"
                                in self.data[branch_name][segment][data_type][metric]
                            ):

                                metric_data = self.data[branch_name][segment][
                                    data_type
                                ][metric]
                                rows = []
                                n_labels = len(metric_data["labels"])
                                for i in range(n_labels):
                                    label = metric_data["labels"][i]
                                    uplift = metric_data["uplift"][i]

                                    # Skip small uplifts for enumerated histograms with many labels
                                    if n_labels > 5 and abs(uplift) < 0.05:
                                        continue

                                    weight = "font-weight:normal;"
                                    if abs(uplift) >= 10:
                                        effect = "Large"
                                        weight = "font-weight:bold;"
                                    elif abs(uplift) >= 5:
                                        effect = "Medium"
                                        weight = "font-weight:bold;"
                                    elif abs(uplift) >= 2:
                                        effect = "Small"
                                    else:
                                        effect = "None"

                                    uplift_str = (
                                        f"+{uplift:.2f}"
                                        if uplift > 0
                                        else f"{uplift:.2f}"
                                    )
                                    uplift_desc = f"{label:<15}: {uplift_str}%"

                                    rows.append(
                                        {
                                            "uplift": uplift_desc,
                                            "effect": effect,
                                            "weight": weight,
                                            "style": f"background:{row_background};",
                                        }
                                    )

                                if rows:
                                    rows[-1]["style"] += "border-bottom-style: solid;"
                                    branches.append(
                                        {
                                            "branch": branch_name,
                                            "style": f"background:{row_background}; border-bottom-style: solid;",
                                            "branch_rowspan": len(rows),
                                            "rows": rows,
                                        }
                                    )

                        if branches:
                            total_rowspan = sum(b["branch_rowspan"] for b in branches)
                            # Get description from any branch that has this metric
                            desc = "Categorical metric"
                            for branch in self.data["branches"]:
                                branch_name = (
                                    branch["name"]
                                    if isinstance(branch, dict)
                                    else branch
                                )
                                if (
                                    branch_name in self.data
                                    and segment in self.data[branch_name]
                                    and data_type in self.data[branch_name][segment]
                                    and metric
                                    in self.data[branch_name][segment][data_type]
                                ):
                                    desc = self.data[branch_name][segment][data_type][
                                        metric
                                    ].get("desc", "Categorical metric")
                                    break

                            categorical_metrics.append(
                                {
                                    "name": metric_name,
                                    "desc": desc,
                                    "style": f"background:{row_background}; border-bottom-style: solid; border-right-style: solid;",
                                    "name_rowspan": total_rowspan,
                                    "branches": branches,
                                }
                            )
                        continue

                    # Generate summary for scalar metrics
                    if data_type == "scalar":
                        datasets = []
                        for branch in self.data["branches"]:
                            branch_name = (
                                branch["name"] if isinstance(branch, dict) else branch
                            )
                            if branch_name == control_name:
                                continue

                            if (
                                branch_name in self.data
                                and segment in self.data[branch_name]
                                and data_type in self.data[branch_name][segment]
                                and metric in self.data[branch_name][segment][data_type]
                            ):

                                metric_data = self.data[branch_name][segment][
                                    data_type
                                ][metric]
                                scalar_count = metric_data.get("count", 0)
                                control_count = metric_data.get("control_count", 0)
                                relative_change = metric_data.get("relative_change", 0)
                                absolute_diff = metric_data.get(
                                    "absolute_difference", 0
                                )

                                relative_change_str = (
                                    f"+{relative_change:.1f}%"
                                    if relative_change > 0
                                    else f"{relative_change:.1f}%"
                                )
                                absolute_diff_str = (
                                    f"+{absolute_diff}"
                                    if absolute_diff > 0
                                    else f"{absolute_diff}"
                                )

                                # Determine effect meaning
                                if abs(relative_change) > 200:
                                    effect_meaning = "Large"
                                    color = (
                                        "font-weight: bold; color: red"
                                        if relative_change > 0
                                        else "font-weight: bold; color: green"
                                    )
                                elif abs(relative_change) >= 100:
                                    effect_meaning = "Medium"
                                    color = (
                                        "font-weight: bold; color: red"
                                        if relative_change > 0
                                        else "font-weight: bold; color: green"
                                    )
                                elif abs(relative_change) >= 50:
                                    effect_meaning = "Small"
                                    color = (
                                        "font-weight: normal; color: red"
                                        if relative_change > 0
                                        else "font-weight: normal; color: green"
                                    )
                                else:
                                    effect_meaning = "None"
                                    color = "font-weight: normal"

                                datasets.append(
                                    {
                                        "branch": branch_name,
                                        "count": scalar_count,
                                        "control_count": control_count,
                                        "relative_change": relative_change_str,
                                        "absolute_diff": absolute_diff_str,
                                        "uplift": f"{relative_change:.1f}",
                                        "effect": effect_meaning,
                                        "color": color,
                                        "style": f"background:{row_background};",
                                    }
                                )

                        if datasets:
                            datasets[-1]["style"] += "border-bottom-style:solid;"
                            # Get description from any branch that has this metric
                            desc = f"Total {metric}"
                            for branch in self.data["branches"]:
                                branch_name = (
                                    branch["name"]
                                    if isinstance(branch, dict)
                                    else branch
                                )
                                if (
                                    branch_name in self.data
                                    and segment in self.data[branch_name]
                                    and data_type in self.data[branch_name][segment]
                                    and metric
                                    in self.data[branch_name][segment][data_type]
                                ):
                                    desc = self.data[branch_name][segment][data_type][
                                        metric
                                    ].get("desc", f"Total {metric}")
                                    break

                            numerical_metrics.append(
                                {
                                    "name": metric,
                                    "desc": desc,
                                    "style": f"background:{row_background}; border-bottom-style:solid; border-right-style:solid;",
                                    "datasets": datasets,
                                    "rowspan": len(datasets),
                                    "is_scalar_metric": True,
                                }
                            )
                        continue

                    # Generate summary for labeled_percentiles metrics
                    if data_type == "labeled_percentiles":
                        datasets = []
                        for branch in self.data["branches"]:
                            branch_name = (
                                branch["name"] if isinstance(branch, dict) else branch
                            )
                            if branch_name == control_name:
                                continue

                            if (
                                branch_name in self.data
                                and segment in self.data[branch_name]
                                and data_type in self.data[branch_name][segment]
                                and metric in self.data[branch_name][segment][data_type]
                            ):
                                metric_data = self.data[branch_name][segment][data_type][metric]

                                # Find label with largest absolute median uplift
                                if "median_uplifts" in metric_data and metric_data["median_uplifts"]:
                                    uplifts = metric_data["median_uplifts"]
                                    labels = metric_data["labels"]

                                    # Find max absolute uplift
                                    max_idx = max(range(len(uplifts)), key=lambda i: abs(uplifts[i]))
                                    max_uplift = uplifts[max_idx]
                                    max_label = labels[max_idx]

                                    uplift_str = (
                                        f"+{max_uplift:.1f}" if max_uplift > 0 else f"{max_uplift:.1f}"
                                    )

                                    # Determine effect size based on uplift magnitude
                                    if abs(max_uplift) >= 10:
                                        effect_meaning = "Large"
                                    elif abs(max_uplift) >= 5:
                                        effect_meaning = "Medium"
                                    elif abs(max_uplift) >= 2:
                                        effect_meaning = "Small"
                                    else:
                                        effect_meaning = "None"

                                    datasets.append({
                                        "branch": branch_name,
                                        "uplift": f"{uplift_str}% ({max_label})",
                                        "effect": effect_meaning,
                                        "style": f"background:{row_background};border-bottom-style:solid;",
                                        "skip_percent": True,  # Don't add % in template
                                    })

                        if datasets:
                            # Get description
                            desc = f"Labeled percentiles: {metric}"
                            for branch in self.data["branches"]:
                                branch_name = (
                                    branch["name"] if isinstance(branch, dict) else branch
                                )
                                if (
                                    branch_name in self.data
                                    and segment in self.data[branch_name]
                                    and data_type in self.data[branch_name][segment]
                                    and metric in self.data[branch_name][segment][data_type]
                                ):
                                    desc = self.data[branch_name][segment][data_type][metric].get("desc", f"Labeled percentiles: {metric}")
                                    break

                            numerical_metrics.append({
                                "name": metric_name,
                                "desc": desc,
                                "style": f"background:{row_background}; border-bottom-style:solid; border-right-style:solid;",
                                "datasets": datasets,
                                "rowspan": len(datasets),
                                "is_labeled_percentiles": True,
                            })
                        continue

                    # Generate summary for quantity_percentiles metrics
                    if data_type == "quantity_percentiles":
                        datasets = []
                        for branch in self.data["branches"]:
                            branch_name = (
                                branch["name"] if isinstance(branch, dict) else branch
                            )
                            if branch_name == control_name:
                                continue

                            if (
                                branch_name in self.data
                                and segment in self.data[branch_name]
                                and data_type in self.data[branch_name][segment]
                                and metric in self.data[branch_name][segment][data_type]
                            ):
                                metric_data = self.data[branch_name][segment][data_type][metric]

                                # Use median uplift for summary
                                if "median_uplift" in metric_data:
                                    median_uplift = metric_data["median_uplift"]

                                    uplift_str = (
                                        f"+{median_uplift:.1f}" if median_uplift > 0 else f"{median_uplift:.1f}"
                                    )

                                    # Determine effect size based on uplift magnitude
                                    if abs(median_uplift) >= 10:
                                        effect_meaning = "Large"
                                    elif abs(median_uplift) >= 5:
                                        effect_meaning = "Medium"
                                    elif abs(median_uplift) >= 2:
                                        effect_meaning = "Small"
                                    else:
                                        effect_meaning = "None"

                                    datasets.append({
                                        "branch": branch_name,
                                        "uplift": f"{uplift_str}% (median)",
                                        "effect": effect_meaning,
                                        "style": f"background:{row_background};border-bottom-style:solid;",
                                        "skip_percent": True,  # Don't add % in template
                                    })

                        if datasets:
                            # Get description
                            desc = f"Quantity percentiles: {metric}"
                            for branch in self.data["branches"]:
                                branch_name = (
                                    branch["name"] if isinstance(branch, dict) else branch
                                )
                                if (
                                    branch_name in self.data
                                    and segment in self.data[branch_name]
                                    and data_type in self.data[branch_name][segment]
                                    and metric in self.data[branch_name][segment][data_type]
                                ):
                                    desc = self.data[branch_name][segment][data_type][metric].get("desc", f"Quantity percentiles: {metric}")
                                    break

                            numerical_metrics.append({
                                "name": metric_name,
                                "desc": desc,
                                "style": f"background:{row_background}; border-bottom-style:solid; border-right-style:solid;",
                                "datasets": datasets,
                                "rowspan": len(datasets),
                                "is_quantity_percentiles": True,
                            })
                        continue

                    # Generate summary for numerical metrics
                    if data_type == "numerical":
                        datasets = []
                        for branch in self.data["branches"]:
                            branch_name = (
                                branch["name"] if isinstance(branch, dict) else branch
                            )
                            if branch_name == control_name:
                                continue

                            if (
                                branch_name in self.data
                                and segment in self.data[branch_name]
                                and data_type in self.data[branch_name][segment]
                                and metric in self.data[branch_name][segment][data_type]
                            ):

                                metric_data = self.data[branch_name][segment][
                                    data_type
                                ][metric]
                                median = f"{metric_data['median']:.1f}"
                                std = f"{metric_data['std']:.1f}"

                                # Calculate uplift using median
                                branch_median = metric_data["median"]

                                # Check if the metric exists in the control branch for this segment
                                if (
                                    control_name in self.data
                                    and segment in self.data[control_name]
                                    and data_type in self.data[control_name][segment]
                                    and metric
                                    in self.data[control_name][segment][data_type]
                                ):
                                    control_median = self.data[control_name][segment][
                                        data_type
                                    ][metric]["median"]
                                else:
                                    # Skip this metric if it doesn't exist in control for this segment
                                    continue
                                uplift = (
                                    (branch_median - control_median)
                                    / control_median
                                    * 100.0
                                )
                                uplift_str = (
                                    f"+{uplift:.1f}" if uplift > 0 else f"{uplift:.1f}"
                                )

                                # Get statistical test results
                                if (
                                    "tests" in metric_data
                                    and "mwu" in metric_data["tests"]
                                ):
                                    pval = metric_data["tests"]["mwu"]["p-value"]
                                    effect_size = metric_data["tests"]["mwu"]["effect"]
                                    effect_meaning = get_rank_biserial_corr_meaning(
                                        effect_size
                                    )
                                    effect = f"{effect_meaning} (p={pval:.2f})"

                                    if pval >= 0.001:
                                        effect = f"None (p={pval:.2f})"
                                        effect_meaning = "None"
                                else:
                                    effect = "No test"
                                    effect_meaning = "None"

                                full_hist_name = None
                                for hist_key in self.data.get("histograms", {}):
                                    if hist_key.endswith(metric):
                                        full_hist_name = hist_key
                                        break

                                higher_is_better = False
                                if (
                                    full_hist_name
                                    and full_hist_name in self.data["histograms"]
                                ):
                                    higher_is_better = self.data["histograms"][
                                        full_hist_name
                                    ].get("higher_is_better", False)
                                if (
                                    effect_meaning == "None"
                                    or effect_meaning == "Small"
                                ):
                                    color = "font-weight: normal"
                                else:
                                    if higher_is_better:
                                        if uplift >= 1.5:
                                            color = "font-weight: bold; color: green"
                                        elif uplift <= -1.5:
                                            color = "font-weight: bold; color: red"
                                        else:
                                            color = "font-weight: normal"
                                    else:
                                        if uplift >= 1.5:
                                            color = "font-weight: bold; color: red"
                                        elif uplift <= -1.5:
                                            color = "font-weight: bold; color: green"
                                        else:
                                            color = "font-weight: normal"

                                datasets.append(
                                    {
                                        "branch": branch_name,
                                        "median": median,
                                        "uplift": uplift_str,
                                        "std": std,
                                        "effect": effect,
                                        "color": color,
                                        "style": f"background:{row_background};",
                                    }
                                )

                        if datasets:
                            datasets[-1]["style"] += "border-bottom-style:solid;"
                            # Get description from any branch that has this metric
                            desc = metric
                            for branch in self.data["branches"]:
                                branch_name = (
                                    branch["name"]
                                    if isinstance(branch, dict)
                                    else branch
                                )
                                if (
                                    branch_name in self.data
                                    and segment in self.data[branch_name]
                                    and data_type in self.data[branch_name][segment]
                                    and metric
                                    in self.data[branch_name][segment][data_type]
                                ):
                                    desc = self.data[branch_name][segment][data_type][
                                        metric
                                    ].get("desc", metric)
                                    break

                            numerical_metrics.append(
                                {
                                    "name": metric,
                                    "desc": desc,
                                    "style": f"background:{row_background}; border-bottom-style:solid; border-right-style:solid;",
                                    "datasets": datasets,
                                    "rowspan": len(datasets),
                                }
                            )

            segments.append(
                {
                    "name": segment,
                    "numerical_metrics": numerical_metrics,
                    "categorical_metrics": categorical_metrics,
                }
            )

        slug = self.data["slug"]
        is_experiment = self.data["is_experiment"]

        if is_experiment:
            startDate = self.data["startDate"]
            endDate = self.data["endDate"]
            channel = self.data["channel"]
        else:
            startDate = None
            endDate = None
            channel = None

        branches = []
        for i in range(len(self.data["input"]["branches"])):
            if is_experiment:
                branchInfo = {"name": self.data["input"]["branches"][i]["name"]}
            else:
                branchInfo = {
                    "name": self.data["input"]["branches"][i]["name"],
                    "startDate": self.data["input"]["branches"][i]["startDate"],
                    "endDate": self.data["input"]["branches"][i]["endDate"],
                    "channel": self.data["input"]["branches"][i]["channel"],
                }
            branches.append(branchInfo)

        context = {
            "slug": slug,
            "is_experiment": is_experiment,
            "startDate": startDate,
            "endDate": endDate,
            "channel": channel,
            "branches": branches,
            "segments": segments,
            "branchlen": len(branches),
        }
        self.doc(t.render(context))

    def createConfigSection(self):
        t = get_template("config.html")
        context = {
            "config": json.dumps(self.data["input"], indent=4),
            "queries": self.data["queries"],
        }
        self.doc(t.render(context))

    def createCDFComparison(self, segment, metric, metric_type):
        t = get_template("cdf.html")

        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        # Access control data using new structure
        if (
            control_name in self.data
            and segment in self.data[control_name]
            and metric_type in self.data[control_name][segment]
            and metric in self.data[control_name][segment][metric_type]
            and "pdf" in self.data[control_name][segment][metric_type][metric]
        ):

            control_data = self.data[control_name][segment][metric_type][metric]
            values_control = control_data["pdf"]["values"]
            cdf_control = control_data["pdf"]["cdf"]

            maxValue = find_value_at_quantile(values_control, cdf_control)
            if maxValue is None:
                return  # Skip chart if no valid quantile found
            values_int = np.around(np.linspace(0, maxValue, 100), 2).tolist()

            datasets = []
            for branch in self.data["branches"]:
                branch_name = branch["name"] if isinstance(branch, dict) else branch

                if (
                    branch_name in self.data
                    and segment in self.data[branch_name]
                    and metric_type in self.data[branch_name][segment]
                    and metric in self.data[branch_name][segment][metric_type]
                    and "pdf" in self.data[branch_name][segment][metric_type][metric]
                ):

                    metric_data = self.data[branch_name][segment][metric_type][metric]
                    values = metric_data["pdf"]["values"]
                    density = metric_data["pdf"]["density"]
                    cdf = metric_data["pdf"]["cdf"]

                    # Smooth out pdf and cdf, and use common X values for each branch.
                    density_int = cubic_spline_smooth(values, density, values_int)
                    cdf_int = cubic_spline_smooth(values, cdf, values_int)

                    dataset = {
                        "branch": branch_name,
                        "cdf": cdf_int,
                        "density": density_int,
                    }

                    datasets.append(dataset)
        else:
            # No control data available, skip this chart
            return

        unit = self.get_metric_unit(metric)
        context = {
            "segment": segment,
            "metric": metric,
            "values": values_int,
            "datasets": datasets,
            "unit": unit,
        }
        self.doc(t.render(context))
        return

    def calculate_uplift_interp(self, quantiles, branch, segment, metric_type, metric):
        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        # Access control data using new structure
        if (
            control_name in self.data
            and segment in self.data[control_name]
            and metric_type in self.data[control_name][segment]
            and metric in self.data[control_name][segment][metric_type]
        ):

            control_data = self.data[control_name][segment][metric_type][metric]
            quantiles_control = control_data.get("quantiles", [])
            values_control = control_data.get("quantile_vals", [])

            if not quantiles_control or not values_control:
                return [[], []]

            [quantiles_control_n, values_control_n] = cubic_spline_prep(
                quantiles_control, values_control
            )
            if len(quantiles_control_n) < 2:
                return [[], []]
            tck = interpolate.splrep(quantiles_control_n, values_control_n, k=1)
            values_control_n = interpolate.splev(quantiles, tck, der=0)

            # Access branch data using new structure
            if (
                branch in self.data
                and segment in self.data[branch]
                and metric_type in self.data[branch][segment]
                and metric in self.data[branch][segment][metric_type]
            ):

                branch_data = self.data[branch][segment][metric_type][metric]
                quantiles_branch = branch_data.get("quantiles", [])
                values_branch = branch_data.get("quantile_vals", [])

                if not quantiles_branch or not values_branch:
                    return [[], []]

                [quantiles_branch_n, values_branch_n] = cubic_spline_prep(
                    quantiles_branch, values_branch
                )
                if len(quantiles_branch_n) < 2:
                    return [[], []]
                tck = interpolate.splrep(quantiles_branch_n, values_branch_n, k=1)
                values_branch_n = interpolate.splev(quantiles, tck, der=0)
            else:
                return [[], []]
        else:
            return [[], []]

        uplifts = []
        diffs = []
        for i in range(len(quantiles)):
            diff = values_branch_n[i] - values_control_n[i]
            uplift = diff / values_control_n[i] * 100
            diffs.append(float(diff))
            uplifts.append(float(uplift))

        return [diffs, uplifts]

    def createUpliftComparison(self, segment, metric, metric_type):
        t = get_template("uplift.html")

        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control
        quantiles = np.around(np.linspace(0.1, 0.99, 99), 2).tolist()

        display_metric = metric

        datasets = []
        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch
            if branch_name == control_name:
                continue

            [diff, uplift] = self.calculate_uplift_interp(
                quantiles, branch_name, segment, metric_type, metric
            )
            dataset = {
                "branch": branch_name,
                "diff": diff,
                "uplift": uplift,
            }
            datasets.append(dataset)

        maxVal = 0
        for x in diff:
            if abs(x) > maxVal:
                maxVal = abs(x)

        maxPerc = 0
        for x in uplift:
            if abs(x) > maxPerc:
                maxPerc = abs(x)

        unit = self.get_metric_unit(metric)
        context = {
            "segment": segment,
            "metric": display_metric,
            "quantiles": quantiles,
            "datasets": datasets,
            "upliftMax": maxPerc,
            "upliftMin": -maxPerc,
            "diffMax": maxVal,
            "diffMin": -maxVal,
            "unit": unit,
        }
        self.doc(t.render(context))

    def createScalarComparison(
        self,
        segment,
        metric,
        template_name="scalar.html",
        value_key="count",
    ):
        """Create comparisons for scalar values.

        Args:
            segment: The segment being analyzed (e.g., 'Windows', 'Mac')
            metric: The metric name (e.g., 'total_crashes')
            template_name: The HTML template to use for rendering
            value_key: The key to use for the scalar value (e.g., 'count')
        """
        t = get_template(template_name)
        control = self.data["branches"][0]

        datasets = []

        for branch in self.data["branches"]:
            # Access scalar data using consistent structure: data[branch][segment]["scalar"][metric]
            metric_data = self.data[branch][segment]["scalar"][metric]

            scalar_value = metric_data.get(value_key, 0)
            desc = metric_data.get("desc", f"Total {metric}")

            dataset = {
                "branch": branch,
                value_key: scalar_value,
                "desc": desc,
                "control": branch == control,
            }

            if branch != control:
                # Access control data using same consistent structure
                control_metric_data = self.data[control][segment]["scalar"][metric]

                control_value = control_metric_data.get(value_key, 0)
                relative_change = metric_data.get("relative_change", 0)
                absolute_diff = metric_data.get("absolute_difference", 0)

                dataset.update(
                    {
                        f"control_{value_key}": control_value,
                        "relative_change": relative_change,
                        "absolute_diff": absolute_diff,
                    }
                )

                # Format the changes for display
                if relative_change > 0:
                    relative_change_str = f"+{relative_change:.1f}%"
                else:
                    relative_change_str = f"{relative_change:.1f}%"

                if absolute_diff > 0:
                    absolute_diff_str = f"+{absolute_diff}"
                else:
                    absolute_diff_str = f"{absolute_diff}"

                dataset.update(
                    {
                        "relative_change_str": relative_change_str,
                        "absolute_diff_str": absolute_diff_str,
                    }
                )

            datasets.append(dataset)

        # Get metric metadata from the first dataset
        first_dataset = datasets[0] if datasets else {}
        metric_title = first_dataset.get("desc", metric)
        value_label = "Count"  # Default label, could be made configurable

        context = {
            "segment": segment,
            "metric": metric,
            "metric_title": metric_title,
            "value_label": value_label,
            "datasets": datasets,
            "control": control,
        }
        self.doc(t.render(context))

    def createMeanComparison(self, segment, metric, metric_type):
        t = get_template("mean.html")

        datasets = []
        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch

            # Access data using new structure: self.data[branch_name][segment][data_type][metric]
            if (
                branch_name in self.data
                and segment in self.data[branch_name]
                and metric_type in self.data[branch_name][segment]
                and metric in self.data[branch_name][segment][metric_type]
            ):

                metric_data = self.data[branch_name][segment][metric_type][metric]

                n_value = int(metric_data.get("n", 0))
                n = f"{n_value:,}"
                median = "{0:.1f}".format(metric_data.get("median", 0))

                if branch_name != control_name:
                    branch_mean = metric_data.get("mean", 0)

                    # Check if the metric exists in the control branch for this segment
                    if (
                        control_name in self.data
                        and segment in self.data[control_name]
                        and metric_type in self.data[control_name][segment]
                        and metric in self.data[control_name][segment][metric_type]
                    ):
                        control_mean = self.data[control_name][segment][metric_type][
                            metric
                        ].get("mean", 0)
                    else:
                        control_mean = 0
                    if control_mean != 0:
                        uplift = (branch_mean - control_mean) / control_mean * 100.0
                        uplift = "{0:.1f}".format(uplift)
                    else:
                        uplift = "0.0"
                else:
                    uplift = ""

                # Handle missing se field - calculate if not present
                se = metric_data.get("se", 0)
                if se == 0 and n_value > 0:
                    # Calculate standard error from std and n
                    std_val = metric_data.get("std", 0)
                    se = std_val / (n_value**0.5) if n_value > 0 else 0
                se = "{0:.1f}".format(se)

                std = "{0:.1f}".format(metric_data.get("std", 0))
                mean = "{0:.1f}".format(metric_data.get("mean", 0))

                dataset = {
                    "branch": branch_name,
                    "mean": mean,
                    "median": median,
                    "uplift": uplift,
                    "n": n,
                    "se": se,
                    "std": std,
                    "control": branch_name == control_name,
                }

                if branch_name != control_name and "tests" in metric_data:
                    for test_name, test_data in metric_data["tests"].items():
                        effect = "{0:.2f}".format(test_data.get("effect", 0))
                        pval = "{0:.2g}".format(test_data.get("p-value", 1.0))
                        dataset[test_name] = {"effect": effect, "pval": pval}

                datasets.append(dataset)

        context = {
            "segment": segment,
            "metric": metric,
            "branches": self.data["branches"],
            "datasets": datasets,
        }
        self.doc(t.render(context))

    def createCategoricalComparison(self, segment, metric, metric_type):
        t = get_template("categorical.html")

        # If the histogram has too many buckets, then only display a
        # set of interesting comparisons instead of all of them.
        indices = set()

        control = self.data["branches"][0]

        n_elem = len(self.data[control][segment][metric_type][metric]["counts"])

        # Get unit from metric config if available
        unit = ""
        for hist_key in self.data.get("histograms", {}):
            hist_name = hist_key.split(".")[-1]
            if hist_name == metric:
                unit = self.data["histograms"][hist_key].get("unit", "")
                break
        if n_elem <= 10:
            indices = set(range(0, n_elem))

        for branch in self.data["branches"]:
            if branch == control:
                continue
            uplift = self.data[branch][segment][metric_type][metric]["uplift"]

            for i in range(len(uplift)):
                # Show categories with > 1% change
                if abs(uplift[i]) > 1:
                    indices.add(i)

        datasets = []
        for branch in self.data["branches"]:
            counts_branch = [
                float(self.data[branch][segment][metric_type][metric]["counts"][i])
                for i in indices
            ]
            datasets.append(
                {
                    "branch": branch,
                    "counts": counts_branch,
                }
            )

            if branch != control:
                uplift = [
                    float(self.data[branch][segment][metric_type][metric]["uplift"][i])
                    for i in indices
                ]
                datasets[-1]["uplift"] = uplift

        labels = [
            self.data[control][segment][metric_type][metric]["labels"][i]
            for i in indices
        ]

        # Build table rows - transposed so each label has branch rows
        indices_list = sorted(list(indices))
        table_rows = []
        for idx, i in enumerate(indices_list):
            branch_rows = []
            for j, dataset in enumerate(datasets):
                count = dataset["counts"][idx]
                # Format count with commas and units for large values
                count_num = int(count)
                if count_num >= 1000000000:
                    count_display = f"{count_num / 1000000000:.1f}B"
                elif count_num >= 1000000:
                    count_display = f"{count_num / 1000000:.1f}M"
                elif count_num >= 1000:
                    count_display = f"{count_num / 1000:.1f}K"
                else:
                    count_display = f"{count_num:,}"

                # Uplift only for non-control branches
                uplift = None
                if j > 0 and len(datasets) > 1 and "uplift" in datasets[1]:
                    uplift = datasets[1]["uplift"][idx]

                branch_rows.append({
                    "branch_name": dataset["branch"],
                    "count": count_display,
                    "uplift": uplift,
                })

            table_rows.append({
                "label": labels[idx],
                "branch_rows": branch_rows,
            })

        context = {
            "labels": labels,
            "datasets": datasets,
            "table_rows": table_rows,
            "branch_names": [d["branch"] for d in datasets],
            "branch_indices": list(range(len(datasets))),
            "metric": metric,
            "segment": segment,
            "unit": unit,
        }
        self.doc(t.render(context))

    def createMetrics(self, segment, metric, metric_type, kind):
        # Perform a separate comparison when data is categorical.
        if kind == "categorical":
            self.createCategoricalComparison(segment, metric, metric_type)
            return

        # Perform specific handling for scalar events
        if kind == "scalar":
            self.createScalarComparison(segment, metric)
            return

        # Add mean comparison
        self.createMeanComparison(segment, metric, metric_type)
        # Add PDF and CDF comparison
        self.createCDFComparison(segment, metric, metric_type)
        # Add uplift comparison
        self.createUpliftComparison(segment, metric, metric_type)

    def createDataTypeMetrics(self, segment):
        """Create metrics organized by data type rather than source section."""
        # Collect all metrics for this segment across all branches and data types
        all_metrics = {"numerical": set(), "categorical": set(), "scalar": set(), "labeled_percentiles": set(), "quantity_percentiles": set()}

        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch
            if branch_name in self.data and segment in self.data[branch_name]:
                for data_type in ["numerical", "categorical", "scalar", "labeled_percentiles", "quantity_percentiles"]:
                    if data_type in self.data[branch_name][segment]:
                        all_metrics[data_type].update(
                            self.data[branch_name][segment][data_type].keys()
                        )

        # Create charts for each data type
        for data_type in ["numerical", "categorical", "scalar", "labeled_percentiles", "quantity_percentiles"]:
            sorted_metrics = sorted(all_metrics[data_type])

            # Group labeled histogram metrics by their parent
            current_parent = None
            for metric in sorted_metrics:
                # Check if this is a labeled histogram metric (contains ':')
                if ":" in metric:
                    parent = metric.split(":")[0]
                    if parent != current_parent:
                        # Add section header for new labeled histogram group
                        current_parent = parent
                        with self.doc.div(
                            klass="labeled-histogram-header",
                            style="background-color: #e3f2fd; padding: 10px; margin: 20px 0 10px 0; border-radius: 5px; border-left: 4px solid #1976d2;",
                        ):
                            self.doc.h3(
                                f"Labeled Histogram: {parent}",
                                style="margin: 0; color: #1976d2;",
                            )
                else:
                    current_parent = None

                self.createDataTypeMetric(segment, metric, data_type)

    def createDataTypeMetric(self, segment, metric, data_type):
        """Create charts for a specific metric of a given data type."""
        # Check if any branch has data for this metric
        has_data = False
        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch
            if (
                branch_name in self.data
                and segment in self.data[branch_name]
                and data_type in self.data[branch_name][segment]
                and metric in self.data[branch_name][segment][data_type]
            ):

                metric_data = self.data[branch_name][segment][data_type][metric]

                # Check if the metric has actual data (not just an empty template)
                if data_type == "numerical" and metric_data.get("n", 0) > 0:
                    has_data = True
                    break
                elif data_type == "categorical" and metric_data.get("sum", 0) > 0:
                    has_data = True
                    break
                elif data_type == "scalar" and "count" in metric_data:
                    has_data = True
                    break
                elif data_type == "labeled_percentiles" and len(metric_data.get("medians", [])) > 0:
                    has_data = True
                    break
                elif data_type == "quantity_percentiles" and metric_data.get("count", 0) > 0:
                    has_data = True
                    break

        if not has_data:
            print(
                f"  WARNING: Skipping chart generation for {segment}/{metric} ({data_type}) - no data available"
            )
            return

        # Wrap content in cell container for proper layout
        with self.doc.div(id=f"{segment}-{metric}", klass="cell"):
            # Add title for metric
            with self.doc.div(klass="title"):
                self.doc(f"({segment}) - {metric}")

            # Check for overflow warning in any branch
            overflow_warning = None
            for branch in self.data["branches"]:
                branch_name = branch["name"] if isinstance(branch, dict) else branch
                if (
                    branch_name in self.data
                    and segment in self.data[branch_name]
                    and data_type in self.data[branch_name][segment]
                    and metric in self.data[branch_name][segment][data_type]
                    and "overflow_warning"
                    in self.data[branch_name][segment][data_type][metric]
                ):
                    overflow_warning = self.data[branch_name][segment][data_type][
                        metric
                    ]["overflow_warning"]
                    break

            # Display overflow warning if present
            if overflow_warning:
                with self.doc.div(
                    style="background-color: #fff3cd; border: 2px solid #856404; padding: 15px; margin: 10px 0; border-radius: 5px;"
                ):
                    with self.doc.p(
                        style="margin: 0; color: #856404; font-weight: bold;"
                    ):
                        self.doc("⚠️ Data Quality Warning")
                    with self.doc.p(style="margin: 5px 0 0 0; color: #856404;"):
                        self.doc(
                            f"{overflow_warning['last_bucket_ratio']*100:.1f}% of all measurements are in the histogram's overflow bucket "
                            f"(value={overflow_warning['last_bucket_value']}). This indicates the histogram's maximum value is too small for the "
                            f"measured data, making the statistical analysis unreliable. Consider using a different metric with appropriate bucket ranges."
                        )

            if data_type == "numerical":
                # Create numerical charts (mean, CDF, uplift, PDF)
                self.createNumericalCharts(segment, metric)
            elif data_type == "categorical":
                # Create categorical charts
                self.createCategoricalCharts(segment, metric)
            elif data_type == "scalar":
                # Create scalar comparison
                self.createScalarCharts(segment, metric)
            elif data_type == "labeled_percentiles":
                # Create labeled percentiles charts
                self.createLabeledPercentilesCharts(segment, metric)
            elif data_type == "quantity_percentiles":
                # Create quantity percentiles charts
                self.createQuantityPercentilesCharts(segment, metric)

    def createNumericalCharts(self, segment, metric):
        """Create all charts for numerical metrics."""
        # Create comprehensive charts like the original system
        # Use "numerical" as the metric_type for all chart methods
        self.createMeanComparison(segment, metric, "numerical")
        self.createCDFComparison(segment, metric, "numerical")
        self.createUpliftComparison(segment, metric, "numerical")

    def createCategoricalCharts(self, segment, metric):
        """Create all charts for categorical metrics."""
        self.createCategoricalComparison(segment, metric, "categorical")

    def createScalarCharts(self, segment, metric):
        """Create all charts for scalar metrics."""
        # Create scalar comparison table and chart
        self.createScalarComparison(segment, metric)

    def createLabeledPercentilesCharts(self, segment, metric):
        """Create charts for labeled_percentiles metrics (median, p75, p95)."""
        self.createLabeledPercentilesComparison(segment, metric, "labeled_percentiles")

    def createLabeledPercentilesComparison(self, segment, metric, metric_type):
        """Create tables comparing percentiles per label across branches."""
        t = get_template("labeled_percentiles.html")

        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        # Get labels from control branch
        if metric not in self.data[control_name][segment][metric_type]:
            return
        metric_data = self.data[control_name][segment][metric_type][metric]
        labels = metric_data["labels"]

        # Get unit from metric config if available
        unit = ""
        for hist_key in self.data.get("histograms", {}):
            hist_name = hist_key.split(".")[-1]
            if hist_name == metric:
                unit = self.data["histograms"][hist_key].get("unit", "")
                break

        datasets = []
        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch
            if metric not in self.data[branch_name][segment][metric_type]:
                continue

            branch_data = self.data[branch_name][segment][metric_type][metric]
            medians = [float(m) for m in branch_data["medians"]]
            p75s = [float(p) for p in branch_data["p75s"]]
            p95s = [float(p) for p in branch_data["p95s"]]
            counts = [int(c) for c in branch_data["counts"]]

            dataset = {
                "branch": branch_name,
                "medians": medians,
                "p75s": p75s,
                "p95s": p95s,
                "counts": counts,
            }

            if branch_name != control_name:
                if "median_uplifts" in branch_data:
                    dataset["median_uplifts"] = [float(u) for u in branch_data["median_uplifts"]]
                if "p75_uplifts" in branch_data:
                    dataset["p75_uplifts"] = [float(u) for u in branch_data["p75_uplifts"]]
                if "p95_uplifts" in branch_data:
                    dataset["p95_uplifts"] = [float(u) for u in branch_data["p95_uplifts"]]

            datasets.append(dataset)

        # Build table rows - transposed so each label has branch rows
        # Use union of all labels across all branches
        all_labels = set()
        for dataset in datasets:
            branch_name = dataset["branch"]
            if branch_name in self.data and segment in self.data[branch_name]:
                branch_data = self.data[branch_name][segment][metric_type].get(metric, {})
                all_labels.update(branch_data.get("labels", []))

        table_rows = []
        for label in sorted(all_labels):
            # Build branch rows for this label
            branch_rows = []
            for j, dataset in enumerate(datasets):
                # Find index of this label in this dataset
                branch_name = dataset["branch"]
                branch_data = self.data[branch_name][segment][metric_type][metric]
                branch_labels = branch_data["labels"]

                if label not in branch_labels:
                    # Skip this label for this branch if it doesn't have data
                    continue

                i = branch_labels.index(label)

                median = dataset["medians"][i]
                p75 = dataset["p75s"][i]
                p95 = dataset["p95s"][i]
                count = dataset["counts"][i]

                # Format values
                if count >= 1000000:
                    count_display = f"{count / 1000000:.1f}M"
                elif count >= 1000:
                    count_display = f"{count / 1000:.1f}K"
                else:
                    count_display = f"{count:,}"
                median_display = f"{median:,.2f}"
                p75_display = f"{p75:,.2f}"
                p95_display = f"{p95:,.2f}"

                # Uplift only for non-control branches
                median_uplift = None
                p75_uplift = None
                p95_uplift = None
                if j > 0:
                    if "median_uplifts" in dataset and i < len(dataset["median_uplifts"]):
                        median_uplift = dataset["median_uplifts"][i]
                    if "p75_uplifts" in dataset and i < len(dataset["p75_uplifts"]):
                        p75_uplift = dataset["p75_uplifts"][i]
                    if "p95_uplifts" in dataset and i < len(dataset["p95_uplifts"]):
                        p95_uplift = dataset["p95_uplifts"][i]

                branch_rows.append({
                    "branch_name": dataset["branch"],
                    "n": count_display,
                    "median": median_display,
                    "p75": p75_display,
                    "p95": p95_display,
                    "median_uplift": median_uplift,
                    "p75_uplift": p75_uplift,
                    "p95_uplift": p95_uplift,
                })

            # Only create table if at least one branch has this label
            if branch_rows:
                table_rows.append({
                    "label": label,
                    "branch_rows": branch_rows,
                })

        context = {
            "labels": labels,
            "datasets": datasets,
            "table_rows": table_rows,
            "branch_names": [d["branch"] for d in datasets],
            "metric": metric,
            "segment": segment,
            "unit": unit,
        }
        self.doc(t.render(context))

    def createQuantityPercentilesCharts(self, segment, metric):
        """Create charts for quantity_percentiles metrics (median, p75, p95)."""
        self.createQuantityPercentilesComparison(segment, metric, "quantity_percentiles")

    def createQuantityPercentilesComparison(self, segment, metric, metric_type):
        """Create a table showing percentiles for quantity metrics."""
        # Get unit from metric config if available
        unit = ""
        for hist_key in self.data.get("histograms", {}):
            hist_name = hist_key.split(".")[-1]
            if hist_name == metric:
                unit = self.data["histograms"][hist_key].get("unit", "")
                break

        # Build table with branches as rows
        branch_rows = []
        control = self.data["branches"][0]
        control_name = control["name"] if isinstance(control, dict) else control

        for branch in self.data["branches"]:
            branch_name = branch["name"] if isinstance(branch, dict) else branch
            if metric not in self.data[branch_name][segment][metric_type]:
                continue

            metric_data = self.data[branch_name][segment][metric_type][metric]

            median = metric_data.get("median", 0)
            p75 = metric_data.get("p75", 0)
            p95 = metric_data.get("p95", 0)
            count = metric_data.get("count", 0)

            # Format values
            if count >= 1000000:
                count_display = f"{count / 1000000:.1f}M"
            elif count >= 1000:
                count_display = f"{count / 1000:.1f}K"
            else:
                count_display = f"{count:,}"
            median_display = f"{median:,.2f}"
            p75_display = f"{p75:,.2f}"
            p95_display = f"{p95:,.2f}"

            # Get uplifts (only for non-control branches)
            median_uplift = metric_data.get("median_uplift") if branch_name != control_name else None
            p75_uplift = metric_data.get("p75_uplift") if branch_name != control_name else None
            p95_uplift = metric_data.get("p95_uplift") if branch_name != control_name else None

            branch_rows.append({
                "branch_name": branch_name,
                "n": count_display,
                "median": median_display,
                "p75": p75_display,
                "p95": p95_display,
                "median_uplift": median_uplift,
                "p75_uplift": p75_uplift,
                "p95_uplift": p95_uplift,
            })

        context = {
            "metric": metric,
            "segment": segment,
            "unit": unit,
            "branch_rows": branch_rows,
        }

        # Render using a simple template (similar to scalar but with percentiles)
        t = get_template("quantity_percentiles.html")
        self.doc(t.render(context))

    def createHistogramMetrics(self, segment):
        for hist in self.data["histograms"]:
            kind = self.data["histograms"][hist]["kind"]
            metric = hist.split(".")[-1]
            metric_title = self.data["histograms"][hist].get("desc", metric)
            metric_id = metric

            with self.doc.div(id=f"{segment}-{metric_id}", klass="cell"):
                # Add title for metric
                with self.doc.div(klass="title"):
                    self.doc(f"({segment}) - {metric_title}")
                self.createMetrics(segment, hist, "histograms", kind)
        return

    def createHTMLReport(self):
        self.createHeader()
        self.createSidebar()

        # Create a summary of results
        self.createSummarySection()

        # Generate charts and tables for each segment and metric
        for segment in self.data["segments"]:
            self.createDataTypeMetrics(segment)

        # Dump the config and queries used for the report
        self.createConfigSection()

        self.endDocument()

        # Prettify the output
        soup = bs(str(self.doc), "html.parser")
        return soup.prettify()
