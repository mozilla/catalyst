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

### 🎨 `just lint-fix`
Automatically fix code formatting issues using Black

### Binary Commands

### 🚀 `just run [ARGS]`
Generate performance reports from experiment configs
```bash
just run --config configs/experiment-name.yaml
```

### 🔍 `just find-latest-experiment [ARGS]`
Find and process latest experiments
```bash
just find-latest-experiment --help
```

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
pageload_event_metrics:
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
- **histograms**: List of histogram metrics to analyze
- **pageload_event_metrics**: Event metrics with max values (min is always 0)

### Example Configs

Available configuration files:
- **`configs/template.yaml`** - Template with default histograms and events (start here!)
- **`configs/`** - Production YAML configurations for existing experiments
