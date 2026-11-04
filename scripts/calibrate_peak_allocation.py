"""Rebuild the peak-anchored allocation calibration from a human reference profile.

Run `keyloom evaluate reference ...` first. Only aggregate ratios and the
profile's identity are written; no chart text or machine paths.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keyloom_mania.paths import APP_HOME
from keyloom_mania.peak_planner import CALIBRATION_RESOURCE, calibrate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, default=APP_HOME / 'evaluation' / 'human-reference.json.gz')
    parser.add_argument('--output', type=Path, default=ROOT / 'keyloom_mania' / 'resources' / CALIBRATION_RESOURCE)
    args = parser.parse_args()
    with gzip.open(args.reference, 'rt', encoding='utf-8') as handle:
        profile = json.load(handle)
    calibration = calibrate(profile)
    args.output.write_text(json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for bucket in calibration['buckets']:
        print(f"{bucket['star']:.2f}★ ({bucket['charts']} charts): {bucket['ratio_median']}")


if __name__ == '__main__':
    main()
