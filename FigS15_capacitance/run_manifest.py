# -*- coding: utf-8 -*-
"""Shared loading layer for the Supplementary Figure 15 package.

Replaces the per-run acquisition logs with a single compact manifest
(``runs.csv``), and replaces the acquisition software's CSV reader with a
local one.  Every number the plotting and checking scripts need comes from
``runs.csv`` plus the response trace itself.
"""

from __future__ import annotations

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_CSV = os.path.join(HERE, "runs.csv")
BASE = os.path.join(HERE, "mask_test_data")


def read_trace(path: str):
    """Read a 4-column tab-separated trace: column 1 = response, column 3 = monitor."""
    resp, mon = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                resp.append(float(parts[1]))
                mon.append(float(parts[3]))
            except ValueError:
                continue
    return np.asarray(resp, dtype=float), np.asarray(mon, dtype=float)


def load_runs():
    """All rows of runs.csv, in file order."""
    with open(RUNS_CSV, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_run(run: str):
    """One run, in the shape the plotting scripts expect, or None if unknown."""
    row = next((r for r in load_runs() if r["run"] == run), None)
    if row is None:
        return None
    csv_path = os.path.join(BASE, row["run"], row["csv"])
    if not os.path.isfile(csv_path):
        return None
    v, _mon = read_trace(csv_path)
    dt = float(row["sampling_interval_ms"]) / 1000.0
    return dict(
        name=row["run"],
        v=np.asarray(v, float).ravel(),
        dt=dt,
        n_settle=int(row["settle_samples"]),
        active=int(row["active_samples"]),
        mask=row["mask"],
        cap=float(row["capacitor_F"]),
        res=float(row["resistor_ohm"]),
        wins=[(float(row["light_on_s"]), float(row["light_off_s"]))],
        ptp=float(row["ptp_V"]),
        laser=float(row["laser_current_mA"]),
        gate=float(row["gate_amp_V"]),
        valid=row["valid"],
        tau_s=float(row["tau_s"]),
        r2=float(row["r2"]),
        # run directories are timestamps, so the name orders them chronologically
        mtime=0.0,
        order=row["run"],
    )
