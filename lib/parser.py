import requests
import json
import yaml
import sys
import os
import datetime

def checkForLocalFile(filename):
  try:
    with open(filename, 'r') as f:
      data = json.load(f)
      return data
  except:
    return None

def loadProbeIndex():
  filename=os.path.join(os.path.dirname(__file__), "probe-index.json")
  data = checkForLocalFile(filename)
  return data

def annotateMetrics(config):
  probeIndex = loadProbeIndex()
  annotateHistograms(config, probeIndex)
  annotatePageloadEventMetrics(config, probeIndex)

def annotatePageloadEventMetrics(config, probeIndex):
  event_schema = probeIndex["glean"]["perf_page_load"]["extra_keys"]

  event_metrics = config['pageload_event_metrics'].copy()
  config['pageload_event_metrics'] = {}

  for metric in event_metrics:
    config['pageload_event_metrics'][metric] = {}
    if metric in event_schema:
      config['pageload_event_metrics'][metric]["desc"] = event_schema[metric]["description"]
      config['pageload_event_metrics'][metric]["min"] = event_metrics[metric][0]
      config['pageload_event_metrics'][metric]["max"] = event_metrics[metric][1]
    else:
      print(f"ERROR: {metric} not found in pageload event schema.") 
      sys.exit(1)

def annotateHistograms(config, probeIndex):
  histograms = config['histograms'].copy()
  config['histograms'] = {}
  for i,hist in enumerate(histograms):
    config['histograms'][hist] = {}
    hist_name = hist.split('.')[-1]

    # Annotate legacy probe.
    if hist_name.upper() in probeIndex["legacy"]:
      schema = probeIndex["legacy"][hist_name.upper()]
      config['histograms'][hist]["glean"] = False
      config['histograms'][hist]["desc"] = schema["description"]
      config['histograms'][hist]["available_on_desktop"] = True
      config['histograms'][hist]["available_on_android"] = False
      kind = schema["details"]["kind"]
      print(hist, kind)
      if kind=="categorical" or kind=="boolean" or kind=="enumerated":
        config['histograms'][hist]["kind"] = "categorical"
        if "labels" in schema["details"]:
          config['histograms'][hist]["labels"] = schema["details"]["labels"]
        elif kind=="boolean":
          config['histograms'][hist]["labels"] = ["no", "yes"]
        elif "n_buckets" in schema["details"]:
          n_buckets = schema["details"]["n_buckets"]
          config['histograms'][hist]["labels"] = list(range(0, n_buckets))
      else:
        config['histograms'][hist]["kind"] = "numerical"


    # Annotate glean probe.
    elif hist_name in probeIndex["glean"]:
      schema = probeIndex["glean"][hist_name]
      config['histograms'][hist]["glean"] = True
      config['histograms'][hist]["desc"] = schema["description"]
  
      # Mark if the probe is available on desktop or mobile.
      config['histograms'][hist]["available_on_desktop"] = False
      config['histograms'][hist]["available_on_android"] = False

      if "gecko" in schema["repos"]:
        config['histograms'][hist]["available_on_desktop"] = True
        config['histograms'][hist]["available_on_android"] = True
      elif "fenix" in schema["repos"]:
        config['histograms'][hist]["available_on_android"] = True
      elif "desktop" in schema["repos"]:
        config['histograms'][hist]["available_on_desktop"] = True

      # Only support timing distribution types for now.
      if schema["type"] == "timing_distribution":
        config['histograms'][hist]["kind"] = "numerical"
      else:
        type=schema["type"]
        print(f"ERROR: Type {type} for {hist_name} not currently supported.") 
        sys.exit(1)

      # Use the high and low values from the legacy mirror as bounds.
      if "telemetry_mirror" in probeIndex["glean"][hist_name]:
        legacy_mirror = probeIndex["glean"][hist_name]["telemetry_mirror"]
        high = probeIndex["legacy"][legacy_mirror]["details"]["high"]
        config['histograms'][hist]['max'] = high

    else:
      print(f"ERROR: {hist_name} not found in histograms schema.") 
      sys.exit(1)

def retrieveNimbusAPI(dataDir, slug, skipCache):
  filename = f"{dataDir}/{slug}-nimbus-API.json"
  if skipCache:
    values = None
  else:
    values = checkForLocalFile(filename)
  if values is not None:
    print(f"Using local config found in {filename}")
    return values

  url=f'https://experimenter.services.mozilla.com/api/v7/experiments/{slug}/'
  print(f"Loading nimbus API from {url}")
  response = requests.get(url)
  if response.ok:
    values = response.json()
    with open(filename, 'w') as f:
        json.dump(values, f, indent=2)
    return values
  else:
    print(f"Failed to retrieve {url}: {response.status_code}")
    sys.exit(1)

# We only care about a few values from the API.
# Specifically, the branch slugs, channels (prioritized) and start/end dates.
def extractValuesFromAPI(api):
  values = {}
  values["startDate"] = api["startDate"]
  values["endDate"] = api["endDate"]

  # Some experiments can use multiple channels, so select
  # the channel based on the following priority: release > beta > nightly
  if "release" in api["channels"]:
    channel = "release"
  elif "beta" in api["channels"]:
    channel = "beta"
  elif "nightly" in api["channels"]:
    channel = "nightly"
  else:
    available_channels = ", ".join(api.get("channels", []))
    raise ValueError(f"No supported channel found. Available channels: [{available_channels}].")
    
  values["channel"] = channel
  values["isRollout"] = api["isRollout"]

  if values["endDate"] is None:
    now = datetime.datetime.now();
    values["endDate"] = now.strftime('%Y-%m-%d')

  values["branches"] = []

  # If a referenceBranch is defined, or a branch named "control"
  # exists, set it as the first element.
  controlBranch = None
  if "targeting" in api and "referenceBranch" in api["targeting"]:
    controlBranch = values["targeting"]["referenceBranch"]
  elif any(branch["slug"] == "control" for branch in api["branches"]):
    controlBranch = "control"

  if controlBranch:
    values["branches"].append({'name': controlBranch})

  for branch in api["branches"]:
    if branch["slug"] == controlBranch:
      continue
    values["branches"].append({'name': branch["slug"]})

  return values

def parseNimbusAPI(dataDir, slug, skipCache):
  apiValues = retrieveNimbusAPI(dataDir, slug, skipCache)
  return extractValuesFromAPI(apiValues)

def parseConfigFile(configFile):
  with open(configFile, "r") as configData:
    config = json.load(configData)
    configData.close()

  if "branches" in config:
    config["is_experiment"] = False
  else:
    config["is_experiment"] = True

  return config
