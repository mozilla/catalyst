# Catalyst - Telemetry Performance Reports

## Overview

Catalyst is a python project designed for analyzing and generating performance reports based on telemetry data with a heavy focus on Nimbus experiments.

## Dependencies

- **Python 3.10+**  
  Make sure Python 3.10 or above is installed on your system.
- [**Google Cloud SDK**](https://cloud.google.com/sdk/docs/install)  
  Install and set up the Cloud SDK, then setup your Google account:
  ```bash
  gcloud config set project mozdata
  ```

## Usage

1.  Authenticate yourg Google account:
    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

2.  Setup venv and install the Python dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Create a config.  Examples can be found in `configs/`

4. Run:
   ```bash
   python generate-perf-report --config {experiment config}
   ```

5. Open the report that is created at `reports/{slug}.html`
