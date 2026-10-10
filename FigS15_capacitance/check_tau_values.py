#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent verification of the decay time constants of Supplementary Figure 15
(Supplementary Fig. 15).

For every run listed in ``runs.csv`` this script re-reads the response trace and
re-derives the time constant from scratch, then compares the result with

  * the value tabulated in ``runs.csv``, and
  * the value printed in the figure legend.

The estimator is the one used to draw the figure: a single-exponential fit of
|V - V_dark| after light-off, with the asymptotic level taken from the 1 s
before light-on, fitted linearly in the logarithmic domain.

Nothing is written.  Exits non-zero if any comparison fails.

    python check_tau_values.py
    python check_tau_values.py --tol 0.02

Dependencies: numpy only.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_manifest import BASE, load_runs, read_trace          # noqa: E402

#: tau values printed in the figure legend, in seconds
FIGURE_LEGEND = {
    ("100 nF", "1111"): 6.98,
    ("100 nF", "0000"): 7.02,
    ("10 nF", "1111"): 0.738,
    ("10 nF", "0000"): 0.727,
    ("1 nF", "1111"): 0.155,
    ("1 nF", "0000"): 0.154,
}


def tau_of(v, dt, n_settle, active):
    """(tau, R^2 of the log-linear fit, step amplitude)."""
    i0 = n_settle
    i1 = min(i0 + active, v.size - 1)
    pre = v[max(0, i0 - int(round(1.0 / dt))):i0]
    v_dark = float(np.mean(pre))
    noise = float(np.std(pre))
    step = float(np.mean(v[max(i1 - 2, 0):i1 + 1])) - v_dark

    t = (np.arange(v.size) - i1) * dt
    y = np.abs(v - v_dark)
    idx = np.flatnonzero((y > max(4.0 * noise, 0.004)) & (np.arange(v.size) >= i1))
    if idx.size:
        brk = np.flatnonzero(np.diff(idx) > 1)
        if brk.size:
            idx = idx[:brk[0] + 1]
    if idx.size < 6:
        return float("nan"), float("nan"), step

    a, b = np.polyfit(t[idx], np.log(y[idx]), 1)
    pred = a * t[idx] + b
    ss = 1 - np.sum((np.log(y[idx]) - pred) ** 2) / max(
        np.sum((np.log(y[idx]) - np.log(y[idx]).mean()) ** 2), 1e-30)
    return float(-1.0 / a), float(ss), step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="relative tolerance (default: 0.01)")
    args = ap.parse_args(argv)

    rows = load_runs()
    if not rows:
        print("runs.csv is empty")
        return 1

    print("=" * 96)
    print("Supplementary Figure 15 -- independent re-derivation of the decay time constants")
    print("=" * 96)
    print("%-8s %-5s %-18s %11s %11s %11s %9s %6s %s"
          % ("cap", "mask", "run", "tau (csv)", "tau (refit)", "tau (legend)",
             "R^2", "valid", "status"))

    ok_all = True
    for r in rows:
        v, _mon = read_trace(os.path.join(BASE, r["run"], r["csv"]))
        tau_fit, r2, _step = tau_of(
            np.asarray(v, float).ravel(),
            float(r["sampling_interval_ms"]) / 1000.0,
            int(r["settle_samples"]), int(r["active_samples"]))

        tau_csv = float(r["tau_s"])
        legend = FIGURE_LEGEND.get((r["capacitor"], r["mask"]))
        ok_csv = abs(tau_fit - tau_csv) <= args.tol * abs(tau_csv)
        ok_leg = legend is None or abs(tau_fit - legend) <= 5 * args.tol * abs(legend)
        ok_all &= ok_csv and ok_leg
        status = "PASS" if (ok_csv and ok_leg) else (
            "FAIL(csv)" if not ok_csv else "FAIL(legend)")
        print("%-8s %-5s %-18s %11.5f %11.5f %11s %9.5f %6s %s"
              % (r["capacitor"], r["mask"], r["run"], tau_csv, tau_fit,
                 "%.3f" % legend if legend else "-", r2, r["valid"] or "-", status))

    print("-" * 96)
    ok_015 = (abs(FIGURE_LEGEND[("1 nF", "1111")] - 0.15) < 0.01
              and abs(FIGURE_LEGEND[("1 nF", "0000")] - 0.15) < 0.01)
    ok_073 = (abs(FIGURE_LEGEND[("10 nF", "1111")] - 0.73) < 0.01
              and abs(FIGURE_LEGEND[("10 nF", "0000")] - 0.73) < 0.01)
    ratio_100_10 = FIGURE_LEGEND[("100 nF", "1111")] / FIGURE_LEGEND[("10 nF", "1111")]
    ratio_10_1 = FIGURE_LEGEND[("10 nF", "1111")] / FIGURE_LEGEND[("1 nF", "1111")]
    print("Consistency with the rounded values quoted elsewhere:")
    print("  1 nF  tau = 0.154-0.155 s  -> 'approximately 0.15 s'    %s"
          % ("OK" if ok_015 else "MISMATCH"))
    print(" 10 nF  tau = 0.727-0.738 s  -> 'approximately 0.73 s'    %s"
          % ("OK" if ok_073 else "MISMATCH"))
    print("  100/10 ratio = %.2f         -> 'approximately ten times' %s"
          % (ratio_100_10, "OK" if abs(ratio_100_10 - 10) < 1.0 else "MISMATCH"))
    print("   10/1  ratio = %.2f         -> 'not ten times'            %s"
          % (ratio_10_1, "OK" if ratio_10_1 < 8 else "MISMATCH"))
    print("-" * 96)
    print("overall: %s" % ("ALL TAU VALUES VERIFIED" if ok_all else "SOME CHECKS FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
