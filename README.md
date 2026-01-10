# Catalyst - Telemetry Performance Reports

## Overview

Catalyst is a python project designed for analyzing and generating performance reports based on telemetry data with a heavy focus on Nimbus experiments.

## Dependencies

- **Python 3.10+**  
  Make sure Python 3.10 or above is installed on your system.
- **Just** (task runner)  
  Install Just for cross-platform task management:
  ```bash
  # macOS
  brew install just
  
  # Windows (Chocolatey)
  choco install just
  
  # Rust/Cargo
  cargo install just
  ```
- [**Google Cloud SDK**](https://cloud.google.com/sdk/docs/install)  
  Install and set up the Cloud SDK, then setup your Google account:
  ```bash
  gcloud config set project mozdata
  ```

## Quick Start

1. **Authenticate your Google account:**
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. **Install dependencies:**
   ```bash
   just install
   ```

3. **Create or use existing configs:**
   ```bash
   # Start with the template (includes default histograms and events)
   cp configs/template.yaml configs/my-experiment.yaml
   
   # Edit the config file to match your experiment
   # See Configuration section below for details
   ```

4. **Generate performance report:**
   ```bash
   # Simple syntax (recommended)
   just run configs/my-experiment.yaml

   # Or using --config flag
   just run --config configs/my-experiment.yaml
   ```

5. **Open the generated report:** `reports/{slug}.html`

## Available Commands

### 📦 `just install`
Install all dependencies (both production and development)

### 🧪 `just test` 
Run the complete test suite including:
- Unit tests (21 comprehensive tests)
- Coverage reporting (HTML + terminal output)

### 🎨 `just lint`
Automatically fix code formatting issues using Black

### Binary Commands

### 🚀 `just run [ARGS]`
Generate performance reports from experiment configs
```bash
# Simple syntax
just run configs/experiment-name.yaml

# Or with --config flag
just run --config configs/experiment-name.yaml

# With additional options
just run configs/experiment-name.yaml --skip-cache
```

### 🔍 `just find-latest-experiment [ARGS]`
Find and process latest experiments (automatically includes CPU time percentiles)
```bash
just find-latest-experiment index.html failures.json
```

**Default metrics included:**
- Memory (total)
- Page load metrics (FCP, load time, LCP)
- CPU time per process type (percentiles: median, p75, p95)
- Crash events
- Pageload events (FCP, LCP, load time, response time)

### 📄 `just update-index [ARGS]`
Update experiment index files
```bash
just update-index index.html
```

### 🔧 `just update-probe-index [ARGS]`
Update probe index from telemetry schemas
```bash
just update-probe-index
```


## Configuration

Catalyst uses **YAML configuration files** to define experiments and metrics.

### Getting Started with Template

Start with the provided template that includes default histograms and events:

```bash
cp configs/template.yaml configs/my-experiment.yaml
```

### Configuration Format

```yaml
slug: experiment-name
segments:
  - Windows
  - Linux
  - Mac
histograms:
  - payload.histograms.memory_total
  - metrics.timing_distribution.performance_pageload_fcp
events:
  # Include crash events to track total crash counts for each experiment branch
  - crash
  
  # Include pageload events with custom metric configurations
  - pageload:
      fcp_time:
        max: 20000
      lcp_time:
        max: 30000
      load_time:
        max: 25000
```

### Key Configuration Fields

- **slug**: Experiment identifier
- **segments**: Target platforms/segments
- **histograms**: List of histogram metrics to analyze (see Histogram Configuration below)
- **events**: Event metrics configuration (replaces legacy pageload_event_metrics)
  - **crash**: Simple crash event tracking
  - **pageload**: Pageload event metrics with max values (min is always 0)

### Histogram Configuration

Histograms can be specified as a simple list or as a dictionary with additional properties:

**Simple list format:**
```yaml
histograms:
  - metrics.timing_distribution.performance_pageload_fcp
  - metrics.custom_distribution.networking_http_3_upload_throughput
```

**Dictionary format with properties:**
```yaml
histograms:
  metrics.custom_distribution.networking_http_3_upload_throughput:
    higher_is_better: true
  metrics.timing_distribution.performance_pageload_fcp:
    higher_is_better: false  # default
```

### Labeled Counter Metrics

`labeled_counter` metrics (like `power_cpu_time_per_process_type_ms`) require an **aggregate mode** to specify how to analyze the per-label data:

**Available aggregate modes:**

**1. `aggregate: sum`** - Sum all values per label (for event counters)
```yaml
histograms:
  - metrics.labeled_counter.javascript_gc_slice_was_long:
      aggregate: sum
```
- Shows: Total counts per label in categorical bar charts
- Layout: Individual tables per label, branches as rows
- Use for: Event counters where you want to see total occurrences

**2. `aggregate: percentiles`** - Calculate percentiles per label (for measurements)
```yaml
histograms:
  - metrics.labeled_counter.power_cpu_time_per_process_type_ms:
      aggregate: percentiles
```
- Shows: Median, p75, p95, and their individual uplifts per label
- Layout: Individual tables per label, branches as rows
- Use for: Understanding distribution characteristics (median, tails)

### Event Configuration Options

The `events` section supports flexible configuration:

**Simple crash tracking:**
```yaml
events:
  - crash
```

**Pageload events with defaults:**
```yaml
events:
  - pageload  # Uses defaults: fcp_time, lcp_time, load_time (all max: 30000)
```

**Custom pageload metrics:**
```yaml
events:
  - pageload:
      fcp_time:
        max: 25000
      custom_metric:
        max: 15000
```

**Both crash and pageload events:**
```yaml
events:
  - crash
  - pageload:
      fcp_time:
        max: 20000
```

### Advanced Configuration Options

**Custom Conditions:**
Add custom SQL conditions to filter your data:
```yaml
custom_conditions:
  - "country_code = 'US'"
  - "app_version >= '100.0'"
```

**Prerequisite CTEs:**
Define reusable SQL CTEs for complex queries:
```yaml
prerequisite_ctes:
  - name: "filtered_users"
    query: "SELECT client_id FROM table WHERE condition = true"
  - name: "cohort_data"
    query: "SELECT * FROM other_table WHERE date >= '2024-01-01'"
```

**Sample Percentage:**
Control data sampling for faster processing (only applies to histograms):
```yaml
sample_pct: 10  # Use 10% of available data
```

**Parallel Query Threads:**
Control the number of parallel BigQuery threads:
```yaml
max_parallel_queries: 8  # Default is 4
```

### Example Configs

Available configuration files:
- **`configs/template.yaml`** - Template with default histograms and events (start here!)
- **`configs/`** - Production YAML configurations for existing experiments
