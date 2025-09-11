#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lib.parser import parseConfigFile
import glob

print("Testing real config files...")
config_files = glob.glob("configs/*.yaml")
success_count = 0

for config_file in config_files:
    try:
        config = parseConfigFile(config_file)
        print(f"✅ {config_file}")
        success_count += 1
    except Exception as e:
        print(f"❌ {config_file}: {e}")

print(f"Successfully parsed {success_count}/{len(config_files[:5])} config files")

if success_count == 0:
    exit(1)
