# Telemetry Perf Reports

## Overview

Catalyst is a python project designed for analyzing and generating performance reports based on telemetry data with a heavy focus on Nimbus experiments.

## Dependencies

Install Python and the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) to access the `gcloud` cli and that you are authenticated with a project defined (e.g. `gcloud config set project mozdata`).

## Usage

1.  Setup venv and install the Python dependencies:
    `python -m venv venv`
    `source venv/bin/activate`
    `pip install -r requirements.txt`

2. Run ```python generate-perf-report --config {experiment config}```

3. Report should be available at reports/{slug}.html
