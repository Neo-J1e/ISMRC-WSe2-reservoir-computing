#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raw-acquisition provenance check for Supplementary Note 6.

Verifies three links in the chain, from the raw measurement files up to the
values printed in the Supplementary Information:

  1. every file listed in ``raw_acquisition_D2DC2C/清单_SHA256.csv`` matches its
     recorded SHA-256 and byte size;
  2. the derived response matrices (``traces.npz`` and
     ``c4d4/C4D4_原始响应.npz``) are identical, element for element, to column 2
     of the corresponding raw CSV;
  3. the Device 4 calibration table is exactly the 160-point light-on window
     (raw indices 750-909) of the matching raw trace.

Nothing is written.  Exit code is non-zero if any check fails.

    python check_raw_provenance.py
    python check_raw_provenance.py --raw-root /path/to/raw_acquisition_D2DC2C
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data"
MANIFEST_NAME = "清单_SHA256.csv"
RC_SUBDIR = "raw_acquisition_D2DC2C"

# the light-on window of every raw trace: 30.0-36.4 s at a 40 ms saved interval
LIGHT_ON_START = 750
LIGHT_ON_POINTS = 160


def _manifest(raw_root: Path) -> pd.DataFrame:
    path = raw_root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            "manifest not found: %s\n"
            "Pass --raw-root pointing at the folder that holds %s."
            % (path, MANIFEST_NAME))
    return pd.read_csv(path, dtype=str)


def check_manifest(raw_root: Path, man: pd.DataFrame) -> bool:
    print("-" * 88)
    print("1. SHA-256 manifest of the raw acquisition bundle")
    print("-" * 88)
    ok = missing = bad = 0
    problems = []
    for row in man.itertuples():
        path = raw_root / row.relative_path.replace("\\", "/")
        if not path.is_file():
            missing += 1
            problems.append("MISSING  %s" % row.relative_path)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row.sha256:
            bad += 1
            problems.append("HASH     %s" % row.relative_path)
        elif path.stat().st_size != int(row.bytes):
            bad += 1
            problems.append("SIZE     %s" % row.relative_path)
        else:
            ok += 1
    print("files listed   : %d" % len(man))
    print("verified       : %d" % ok)
    print("missing        : %d" % missing)
    print("hash / size    : %d" % bad)
    for line in problems[:10]:
        print("   %s" % line)
    if len(problems) > 10:
        print("   ... and %d more" % (len(problems) - 10))
    return not problems


def check_traces(data_root: Path, raw_root: Path, man: pd.DataFrame) -> bool:
    print()
    print("-" * 88)
    print("2. derived response matrices versus column 2 of the raw CSV")
    print("-" * 88)
    fd = data_root / "five_device"
    z = dict(np.load(fd / "traces.npz"))
    z.update(dict(np.load(fd / "c4d4" / "C4D4_原始响应.npz")))

    checked = skipped = 0
    problems = []
    for row in man.itertuples():
        if not row.relative_path.lower().endswith(".csv"):
            continue
        if not row.device_folder or not row.cycle or not row.mask:
            skipped += 1
            continue
        key = "%s_%s_%s" % (row.device_folder, row.cycle, row.mask)
        raw = np.loadtxt(raw_root / row.relative_path.replace("\\", "/"),
                         delimiter="\t")[:, 1]
        if key not in z:
            problems.append("NO KEY   %s" % key)
            continue
        derived = np.asarray(z[key], dtype=float).ravel()
        checked += 1
        if derived.shape != raw.shape:
            problems.append("SHAPE    %s %s vs %s" % (key, derived.shape, raw.shape))
        elif not np.array_equal(derived, raw):
            problems.append("VALUE    %s  max |diff| = %.3e"
                            % (key, float(np.max(np.abs(derived - raw)))))
    print("traces compared: %d" % checked)
    print("rows skipped   : %d  (not a per-trace recording)" % skipped)
    print("mismatches     : %d" % len(problems))
    for line in problems[:10]:
        print("   %s" % line)
    if len(problems) > 10:
        print("   ... and %d more" % (len(problems) - 10))
    return not problems


def check_device4_calibration(data_root: Path, raw_root: Path,
                              man: pd.DataFrame) -> bool:
    print()
    print("-" * 88)
    print("3. Device 4 calibration table versus the light-on window of the raw trace")
    print("-" * 88)
    table = pd.read_csv(data_root / "parameter_variation" /
                        "Device4_10条原始与拟合数据.csv")
    group = man[man["group"] == "01"]
    # select on the internal device id: the manifest spells the display label
    # both "Device 4" and "Device4" depending on the export
    group = group[group["device_folder"] == "C4D3"]
    checked = 0
    problems = []
    for mask in sorted(table["mask"].unique()):
        for cycle in sorted(table["cycle"].unique()):
            sel = group[(group["cycle"] == str(cycle))
                        & (group["mask"] == "%04d" % mask)]
            if sel.empty:
                problems.append("NO ROW   mask %04d cycle %d" % (mask, cycle))
                continue
            raw = np.loadtxt(raw_root / sel.iloc[0].relative_path.replace("\\", "/"),
                             delimiter="\t")[:, 1]
            window = raw[LIGHT_ON_START:LIGHT_ON_START + LIGHT_ON_POINTS]
            sub = table[(table["mask"] == mask) & (table["cycle"] == cycle)]
            measured = sub["measured_V"].to_numpy()
            checked += 1
            if len(window) != len(measured):
                problems.append("LENGTH   mask %04d cycle %d  %d vs %d"
                                % (mask, cycle, len(window), len(measured)))
            elif not np.array_equal(window, measured):
                problems.append("VALUE    mask %04d cycle %d  max |diff| = %.3e"
                                % (mask, cycle, float(np.max(np.abs(window - measured)))))
    print("traces compared: %d" % checked)
    print("mismatches     : %d" % len(problems))
    for line in problems[:10]:
        print("   %s" % line)
    return not problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify the raw acquisition data against the derived data.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="the Note 6 data/ directory (default: ./data)")
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="folder holding the raw bundle (default: <data-root>/%s)"
                         % RC_SUBDIR)
    args = ap.parse_args(argv)

    data_root = args.data_root.resolve()
    raw_root = (args.raw_root.resolve() if args.raw_root
                else data_root / RC_SUBDIR)
    print("data root : %s" % data_root)
    print("raw root  : %s" % raw_root)
    print()
    if not raw_root.is_dir():
        print("raw acquisition folder not found: %s" % raw_root)
        print("Supplementary Note 6 is still fully reproducible without it")
        print("(the derived data are bundled), so this check is skipped.")
        return 0

    man = _manifest(raw_root)
    results = [check_manifest(raw_root, man),
               check_traces(data_root, raw_root, man),
               check_device4_calibration(data_root, raw_root, man)]
    print()
    print("=" * 88)
    ok = all(results)
    print("overall: %s" % ("RAW PROVENANCE VERIFIED" if ok else "PROVENANCE CHECK FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
