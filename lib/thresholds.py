# Bootstrap iterations per `bootstrap_mode` (experiment YAML). More
# iterations make CI bounds more stable between runs, not narrower: width is
# set by how many clients contribute.
BOOTSTRAP_ITERATIONS = {
    "fast": 200,
    "production": 1000,
    "high_precision": 5000,
}
DEFAULT_BOOTSTRAP_MODE = "production"

# Temporary per-report flag (`enable_bootstrap_ci` in the experiment YAML)
# gating everything the bootstrap CI upgrade changes. Remove once trusted.
DEFAULT_ENABLE_BOOTSTRAP_CI = False

# Minimum observations per branch/segment before a histogram is reported.
MIN_SAMPLE_SIZE = {
    "timing_distribution": 50000,
    "memory_distribution": 25000,
    "categorical": 5000,
    "scalar": 1000,
}
DEFAULT_MIN_SAMPLE_SIZE = 1000


def get_min_sample_size(distribution_type: str, kind: str) -> int:
    key = kind if kind in ("categorical", "scalar") else distribution_type
    return MIN_SAMPLE_SIZE.get(key, DEFAULT_MIN_SAMPLE_SIZE)
