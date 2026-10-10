#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Note 2 -- Effective dimensionality and memory capacity
of the measured reservoir states.

Regenerates every number quoted in Supplementary Note 2 and plotted in
Supplementary Figure 5.

No absolute path is required: all inputs are resolved relative to this
file.  Use --data-root to point at a different copy of the data tree.

    python analyze_effective_rank_mc.py                # analyse + self-check
    python analyze_effective_rank_mc.py --no-check     # analyse only

Outputs are written to ./outputs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data"
DEFAULT_OUT_DIR = HERE / "outputs"

# --------------------------------------------------------------------------- #
# fixed analysis constants (identical to the values used for the manuscript)
# --------------------------------------------------------------------------- #
MASKS = [format(i, "04b") for i in range(16)]
MASKED_WORDS = ["0001", "0010", "0011", "0100", "0101", "0110", "0111",
                "1000", "1001", "1010", "1011", "1100", "1101", "1110"]
FIXED_WORDS = ["0000", "1111"]        # mask words that keep the gate fixed

N_NODES = 16        # 4 mask bits x 4 input cycles
SLOT = 10           # samples per virtual node (dt = theta / 10)
T_STEPS = 2000      # input steps of the classification waveform
CYCLE = 8           # the input waveform is periodic with period 8 steps
DMAX = 24           # delays 1 ... 24, i.e. three input cycles
LAMBDA = 1e-3       # ridge regularisation
USE_TANH = False    # True -> tanh preprocessing of Fig. 3, False -> raw rescaled states

# session, mask-interval duration (ms), gate amplitude (V)
# This is the measurement behind Supplementary Fig. 5, for which the SI states
# "a mask-interval duration theta = 400 ms and a gate amplitude of +/-2 V".
SESSIONS = [
    ("2025-01-02-23-08", "400", 2.0),
]

# state matrices of the classification sessions (rows = mask words, 8000 samples each)
# each file holds a single variable `data_all` of shape (n_masks, 8000)
MASKED_MAT = "chip1-classification-20241129.mat"       # 14 time-varying masks, (14, 8000)
NOMASK_MAT = "chip1-classification-20250101.mat"       # fixed gates, (11, 8000); rows 5,6

# values quoted in Supplementary Note 2 (used by the self-check)
REFERENCE = {
    "input_indexed_min": 3.80,
    "input_indexed_max": 7.43,
    "input_indexed_mean": 6.06,
    "mask_indexed_min": 5.01,
    "mask_indexed_max": 8.36,
    "mask_indexed_mean": 6.37,
    "mask_indexed_sd": 1.00,
    "fixed_mask_0000": 3.25,
    "fixed_mask_1111": 3.37,
    "mc_masked_per_cycle": 0.45,
    "mc_masked_per_cycle_sd": 0.08,
    "mc_fixed_0000_per_cycle": -0.35,
    "mc_fixed_1111_per_cycle": -0.39,
    "mc_masked_sum": 3.59,
    "mc_fixed_0000_sum": -2.80,
    "mc_fixed_1111_sum": -3.14,
    "nl_masked": 0.01,
    "nl_fixed": -0.03,
}


# =========================================================================== #
# 1. virtual-node extraction
# =========================================================================== #
def load_recording(root: Path, mask: str, series: str) -> np.ndarray:
    """One recording: tab separated, column 1 = V_OC(t) [V], column 3 = V_g(t) [V]."""
    path = root / mask / series / (series + ".csv")
    if not path.is_file():
        raise FileNotFoundError("recording not found: %s" % path)
    return np.loadtxt(path)


def smooth_gate(vg, pos_pulse, neg_pulse):
    """Round the gate trace and repair intermediate levels (3-point forward fill)."""
    vg_t = np.round(vg).astype(float)
    for r in np.where(np.isnan(vg_t))[0]:
        if r + 1 < len(vg_t):
            vg_t[r] = vg_t[r + 1]
    for _ in range(5):
        for i in range(len(vg_t) - 3):
            if vg_t[i + 1] != neg_pulse and vg_t[i + 1] != pos_pulse:
                if vg_t[i + 2] == neg_pulse:
                    vg_t[i + 1] = neg_pulse
                elif vg_t[i + 2] == pos_pulse:
                    vg_t[i + 1] = pos_pulse
    return vg_t


def refine(voc, pos, bit):
    """Move a node onto the local V_OC extremum: bit 1 -> maximum, bit 0 -> minimum."""
    i = int(round(pos))
    i = max(0, min(i, len(voc) - 2))
    seg = voc[i:i + 3]
    j = int(np.argmax(seg)) if bit == 1 else int(np.argmin(seg))
    return i + j


def extract_masked(voc, vg, bits, pos_pulse, neg_pulse, n_nodes=N_NODES, slot_w=SLOT):
    """16 virtual-node values of one masked recording."""
    vg_t = smooth_gate(vg, pos_pulse, neg_pulse)
    delta = vg_t - np.concatenate([vg_t[1:], [0.0]])
    if_change = delta != 0
    for i in range(len(if_change) - 1):          # drop duplicated transitions
        if if_change[i] and if_change[i + 1]:
            if_change[i + 1] = False
    edges = np.where(if_change)[0].astype(float)
    if len(edges) == 0:
        return None
    pol, ref = True, []                          # refine edges, alternating polarity
    for e in edges:
        ref.append(refine(voc, e, 1 if pol else 0))
        pol = not pol
    ref = sorted(set(ref))
    e0 = ref[0]
    pos = []
    for k in range(n_nodes):                     # equal spacing, snapped to real edges
        p = e0 + k * slot_w
        d = np.abs(np.array(ref) - p)
        j = int(np.argmin(d))
        p2 = ref[j] if d[j] <= 3 else p
        pos.append(refine(voc, p2, bits[k % 4]))
    return np.array(pos)


def extract_fixed(voc, e0, n_nodes=N_NODES, slot_w=SLOT):
    """Fixed mask words have no transition: uniform grid over the same window."""
    return np.array([int(round(e0 + k * slot_w)) for k in range(n_nodes)])


def build_state_matrix(root: Path, amp: float) -> np.ndarray:
    """256 x 16 state matrix of one session (rows = mask word x optical input)."""
    pos_pulse, neg_pulse = amp, -amp
    X = np.full((256, N_NODES), np.nan)
    e0s = []
    for mi, m in enumerate(MASKS):
        bits = [int(c) for c in m]
        for si, s in enumerate(MASKS):
            row = mi * 16 + si
            a = load_recording(root, m, s)
            voc, vg = a[:, 1], a[:, 3]
            if m in ("0000", "1111"):
                continue
            p = extract_masked(voc, vg, bits, pos_pulse, neg_pulse)
            if p is not None and len(p) == N_NODES:
                X[row, :] = voc[np.minimum(p, len(voc) - 1)]
                e0s.append(p[0])
    e0m = float(np.median(e0s))                  # common window of the session
    for mi, m in enumerate(MASKS):
        if m not in ("0000", "1111"):
            continue
        for si, s in enumerate(MASKS):
            a = load_recording(root, m, s)
            X[mi * 16 + si, :] = a[extract_fixed(a[:, 1], e0m), 1]
    return X


# =========================================================================== #
# 2. dimensionality metrics
# =========================================================================== #
def singular_values(X: np.ndarray) -> np.ndarray:
    """Singular values of the centred state matrix, in decreasing order."""
    return np.maximum(np.linalg.svd(X - X.mean(axis=0), compute_uv=False), 0.0)


def eff_rank(X: np.ndarray) -> float:
    """Participation-ratio effective rank (sum s)^2 / sum s^2."""
    s = singular_values(X)
    return float((s.sum() ** 2) / (s ** 2).sum())


def numerical_rank(X: np.ndarray, rel: float = 1e-2) -> int:
    """Number of singular values above rel * sigma_max."""
    s = singular_values(X)
    return int((s > rel * s[0]).sum())


def exact_rank(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X - X.mean(axis=0)))


def mean_abs_corr(X: np.ndarray) -> float:
    C = np.corrcoef(X.T)
    return float(np.abs(C[~np.eye(X.shape[1], dtype=bool)]).mean())


def analyse_dimensions(data_root: Path, out_dir: Path) -> dict:
    print("=" * 88)
    print("TABLE 1  |  effective rank under matched sampling (16 identical slots)")
    print("=" * 88)
    print("%24s %11s %11s %12s %7s %17s"
          % ("session / tau", "fixed 0000", "fixed 1111",
             "masked mean", "ratio", "mask-config r_eff"))

    multi_states = data_root / "multi_states"
    rows, ratios, fixed_all, masked_all, spectra = [], [], [], [], {}
    per_mask_400 = per_input_400 = None

    for session, tau, amp in SESSIONS:
        root = multi_states / session / tau
        X = build_state_matrix(root, amp)
        per_mask = [eff_rank(X[i * 16:(i + 1) * 16]) for i in range(16)]
        f0, f1 = per_mask[0], per_mask[15]
        masked = float(np.mean(per_mask[1:15]))
        ratio = masked / ((f0 + f1) / 2.0)
        div = [eff_rank(X[[i * 16 + s for i in range(16)]]) for s in range(16)]
        ratios.append(ratio)
        fixed_all += [f0, f1]
        masked_all.append(masked)
        spectra[tau] = singular_values(X)
        if tau == "400":
            per_mask_400, per_input_400 = per_mask, div
        print("%24s %11.2f %11.2f %12.2f %7.2f %17.2f"
              % (session[-5:] + " / " + tau + " ms", f0, f1, masked, ratio,
                 float(np.mean(div))))
        rows.append({
            "session": session, "tau_ms": int(tau), "gate_amplitude_V": amp,
            "fixed_0000_eff_rank": f0, "fixed_1111_eff_rank": f1,
            "masked_mean_eff_rank": masked, "masked_over_fixed_ratio": ratio,
            "mask_configuration_eff_rank_mean": float(np.mean(div)),
        })

    print("-" * 88)
    print("fixed mask words  0000 / 1111 : %.2f / %.2f"
          % (fixed_all[0], fixed_all[1]))
    print("masked mask words (14)         : %.2f" % float(np.mean(masked_all)))
    print("masked / fixed ratio           : %.2f" % float(np.mean(ratios)))

    # ---- detailed metrics for the reference session (tau = 400 ms) ----------
    X = build_state_matrix(multi_states / "2025-01-02-23-08" / "400", 2.0)
    print()
    print("numerical rank (sigma > 1e-2 sigma_max) and exact rank, tau = 400 ms:")
    rank_rows = []
    for group, idx in (("fixed", [0, 15]), ("masked", list(range(1, 15)))):
        nr = [numerical_rank(X[i * 16:(i + 1) * 16]) for i in idx]
        er = [exact_rank(X[i * 16:(i + 1) * 16]) for i in idx]
        print("  %s: numerical rank = %.1f [%d, %d]   exact rank = %.1f"
              % (group, float(np.mean(nr)), min(nr), max(nr), float(np.mean(er))))
        rank_rows.append({"group": group,
                          "numerical_rank_mean": float(np.mean(nr)),
                          "numerical_rank_min": int(min(nr)),
                          "numerical_rank_max": int(max(nr)),
                          "exact_rank_mean": float(np.mean(er))})

    Xm = np.vstack([X[i * 16:(i + 1) * 16] for i in range(1, 15)])
    corr_all = mean_abs_corr(X)
    corr_masked = mean_abs_corr(Xm)
    corr_f0 = mean_abs_corr(X[0:16])
    corr_f1 = mean_abs_corr(X[240:256])
    print("  mean |correlation| between nodes: all 256 rows = %.2f, "
          "14 time-varying masks = %.2f, fixed 0000 = %.2f, fixed 1111 = %.2f"
          % (corr_all, corr_masked, corr_f0, corr_f1))

    M = X[[i * 16 + 15 for i in range(16)]]                  # one input, 16 mask words
    s = singular_values(M)
    print("  effective dimension of the 16 x 16 mask-configuration matrix "
          "(mean over the 16 inputs) = %.1f" % float(np.mean(per_input_400)))
    print("  input '1111': r_eff = %.2f, sigma2/sigma1 = %.2f"
          % (eff_rank(M), s[1] / s[0]))
    endp = M[:, -1] - M[:, -1].mean()
    print("  rank of the 16 final-node values alone = %d  "
          "(single endpoint carries no dimensionality)"
          % np.linalg.matrix_rank(endp.reshape(-1, 1)))

    # ---- write per-input / per-mask tables (Supp. Fig. 5a and 5c) ----------
    with open(out_dir / "note2_effrank_by_optical_input.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["optical_input", "eff_rank_400ms"])
        for s_i, m in enumerate(MASKS):
            w.writerow([m, "%.6f" % per_input_400[s_i]])
    with open(out_dir / "note2_effrank_by_mask_word.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mask_word", "eff_rank_400ms", "fixed_gate"])
        for s_i, m in enumerate(MASKS):
            w.writerow([m, "%.6f" % per_mask_400[s_i], int(m in FIXED_WORDS)])
    with open(out_dir / "note2_dimension_table.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return {
        "table1": rows,
        "tau400_per_mask": {m: v for m, v in zip(MASKS, per_mask_400)},
        "tau400_per_input": {m: v for m, v in zip(MASKS, per_input_400)},
        "tau400_rank_summary": rank_rows,
        "mean_abs_corr": {"all": corr_all, "masked": corr_masked,
                          "fixed_0000": corr_f0, "fixed_1111": corr_f1},
        "masked_over_fixed_ratio_mean": float(np.mean(ratios)),
        "masked_over_fixed_ratio_sd": float(np.std(ratios)),
        "spectra_400ms": spectra["400"].tolist(),
    }


# =========================================================================== #
# 3. memory capacity at matched state size
# =========================================================================== #
def load_rows(matlab_dir: Path, name: str) -> np.ndarray:
    """Load one processed state matrix (rows = recordings of 8000 samples).

    Each bundled .mat file stores exactly one variable, `data_all`.  A clear
    error is raised if the file carries a different variable name, so that a
    mismatched file cannot be mistaken for a valid result.
    """
    path = matlab_dir / name
    if not path.is_file():
        raise FileNotFoundError("state matrix not found: %s" % path)
    from scipy.io import loadmat
    mat = loadmat(str(path))
    if "data_all" not in mat:
        keys = sorted(k for k in mat if not k.startswith("__"))
        raise KeyError(
            "%s does not contain the variable 'data_all' (found: %s). "
            "This file is probably a readout-result file rather than a "
            "reservoir state matrix." % (path.name, ", ".join(keys) or "none"))
    return np.asarray(mat["data_all"], dtype=float)


def load_waveform(matlab_dir: Path):
    base = matlab_dir / "20240818"
    laser = np.loadtxt(base / "wave_classification_laser.txt")
    gt = np.loadtxt(base / "wave_classification_gt.txt")
    u = (laser - 7.5) / 7.5                     # 16 levels -> [-1, 1]
    return u, gt


def states_from_rows(rows) -> np.ndarray:
    """(2000, 4 * n_rows) state matrix from n_rows recordings of 8000 samples."""
    X = np.vstack(rows)
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    curve = (X - lo) / (hi - lo)
    if USE_TANH:                                # optional Fig. 3 preprocessing
        curve = 1.0 + np.tanh((curve - 0.5) * 20.0)
    return curve.reshape(X.shape[0], T_STEPS, 4).transpose(1, 0, 2).reshape(T_STEPS, -1)


def ridge(X, y, lam: float = LAMBDA):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)


def mc_curve(S: np.ndarray, target: np.ndarray, dmax: int = DMAX) -> np.ndarray:
    """MC_k for k = 1 ... dmax: test-set R^2 of u(t-k) read from the state at t."""
    out = []
    for d in range(1, dmax + 1):
        t = np.arange(d, S.shape[0])
        X, y = S[t], target[t - d]
        cut = int(0.7 * len(y))
        w = ridge(X[:cut], y[:cut])
        yh = np.hstack([X[cut:], np.ones((len(y) - cut, 1))]) @ w
        ssr = ((y[cut:] - yh) ** 2).sum()
        sst = ((y[cut:] - y[cut:].mean()) ** 2).sum()
        out.append(1.0 - ssr / sst if sst > 0 else 0.0)
    return np.array(out)


def per_cycle(mc: np.ndarray) -> np.ndarray:
    """Average MC_k over each input cycle of CYCLE = 8 delays."""
    return np.array([mc[i * CYCLE:(i + 1) * CYCLE].mean()
                     for i in range(len(mc) // CYCLE)])


def analyse_memory_capacity(data_root: Path, out_dir: Path) -> dict:
    matlab_dir = data_root / "matlab"
    u, gt = load_waveform(matlab_dir)
    masked = load_rows(matlab_dir, MASKED_MAT)      # 14 time-varying mask words
    nomask = load_rows(matlab_dir, NOMASK_MAT)      # rows 5, 6: mask words 0000 and 1111
    fixed_rows = [nomask[4], nomask[5]]

    print()
    print("=" * 88)
    print("TABLE 2  |  memory capacity at matched state size (4 nodes per step)")
    print("=" * 88)
    print("The input waveform has an 8-step period; the MC values below are averaged")
    print("over each input cycle (8 delays), and three cycles are reported.")
    print()

    curves = {m: mc_curve(states_from_rows([masked[i]]), u)
              for i, m in enumerate(MASKED_WORDS)}
    fx = {m: mc_curve(states_from_rows([r]), u) for m, r in zip(FIXED_WORDS, fixed_rows)}
    allc = np.array([per_cycle(curves[m]) for m in MASKED_WORDS])
    tot = np.array([curves[m][:CYCLE].sum() for m in MASKED_WORDS])

    print("one recording on each side (4 nodes per step, identical state size):")
    print("  mask 1010: per-cycle %s, mean %+.2f"
          % (np.round(per_cycle(curves["1010"]), 2), curves["1010"].mean()))
    print("  mask 1111: per-cycle %s, mean %+.2f"
          % (np.round(per_cycle(fx["1111"]), 2), fx["1111"].mean()))
    print()
    print("all %d time-varying mask words, one recording each:" % len(MASKED_WORDS))
    print("  per-cycle mean over masks: %s" % np.round(allc.mean(0), 2))
    print("  spread over masks in cycle 1: %+.2f ... %+.2f"
          % (allc[:, 0].min(), allc[:, 0].max()))
    print("  positive in %d/%d masks; sum over one input cycle: %+.2f (%+.2f ... %+.2f)"
          % (int((tot > 0).sum()), len(MASKED_WORDS), tot.mean(), tot.min(), tot.max()))
    print()
    fc = np.array([per_cycle(fx[m]) for m in FIXED_WORDS])
    print("fixed-gate mask words 0000 and 1111, one recording each:")
    print("  per-cycle: %s (0000), %s (1111); mean %+.2f"
          % (np.round(fc[0], 2), np.round(fc[1], 2), fc.mean()))
    print("  sum over one input cycle: %+.2f (0000), %+.2f (1111)"
          % (fx["0000"][:CYCLE].sum(), fx["1111"][:CYCLE].sum()))
    print()
    print("NOTE  the values quoted in Supplementary Note 2 are those of the FIRST")
    print("      input period (delays 1-8), i.e. the first entry of each per-cycle")
    print("      array above; the later periods are reported for transparency only.")
    s56 = per_cycle(mc_curve(states_from_rows([masked[i]
                                               for i in range(len(MASKED_WORDS))]), u))
    print("reference: all %d masks together (%d nodes per step): per-cycle %s"
          % (len(MASKED_WORDS), 4 * len(MASKED_WORDS), np.round(s56, 2)))
    nlm = np.array([per_cycle(mc_curve(states_from_rows([masked[i]]), u ** 2))
                    for i in range(len(MASKED_WORDS))]).mean(0)
    nlf = np.array([per_cycle(mc_curve(states_from_rows([r]), u ** 2))
                    for r in fixed_rows]).mean(0)
    print("nonlinear capacity (target u(t-k)^2): masked per-cycle %s, "
          "fixed-gate per-cycle %s" % (np.round(nlm, 2), np.round(nlf, 2)))

    with open(out_dir / "note2_memory_capacity.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "mask_word", "cycle", "mc_per_cycle",
                    "mc_sum_first_cycle"])
        for i, m in enumerate(MASKED_WORDS):
            for c, v in enumerate(per_cycle(curves[m])):
                w.writerow(["masked", m, c + 1, "%.6f" % v,
                            "%.6f" % curves[m][:CYCLE].sum()])
        for m in FIXED_WORDS:
            for c, v in enumerate(per_cycle(fx[m])):
                w.writerow(["fixed_gate", m, c + 1, "%.6f" % v,
                            "%.6f" % fx[m][:CYCLE].sum()])

    return {
        "masked_per_cycle_mean": float(allc[:, 0].mean()),
        "masked_per_cycle_sd": float(allc[:, 0].std()),
        "masked_per_cycle_over_3_cycles": allc.mean(0).tolist(),
        "masked_positive_count": int((tot > 0).sum()),
        "masked_n_masks": len(MASKED_WORDS),
        "masked_sum_mean": float(tot.mean()),
        "masked_sum_min": float(tot.min()),
        "masked_sum_max": float(tot.max()),
        "fixed_0000_per_cycle_mean": float(fc[0][0]),
        "fixed_1111_per_cycle_mean": float(fc[1][0]),
        "fixed_per_cycle_over_3_cycles": {"0000": fc[0].tolist(),
                                          "1111": fc[1].tolist()},
        "fixed_0000_sum": float(fx["0000"][:CYCLE].sum()),
        "fixed_1111_sum": float(fx["1111"][:CYCLE].sum()),
        "nonlinear_masked_per_cycle_mean": float(nlm.mean()),
        "nonlinear_fixed_per_cycle_mean": float(nlf.mean()),
        "all_masks_together_per_cycle": s56.tolist(),
    }


# =========================================================================== #
# 4. self-check against the values printed in Supplementary Note 2
# =========================================================================== #
def self_check(dim: dict, mc: dict) -> bool:
    pi = np.array(list(dim["tau400_per_input"].values()))
    pm = dim["tau400_per_mask"]
    varying = np.array([pm[m] for m in MASKED_WORDS])
    checks = [
        ("optical-input-indexed r_eff min", pi.min(), REFERENCE["input_indexed_min"], 0.01),
        ("optical-input-indexed r_eff max", pi.max(), REFERENCE["input_indexed_max"], 0.01),
        ("optical-input-indexed r_eff mean", pi.mean(), REFERENCE["input_indexed_mean"], 0.01),
        ("mask-indexed r_eff min (14 masks)", varying.min(), REFERENCE["mask_indexed_min"], 0.01),
        ("mask-indexed r_eff max (14 masks)", varying.max(), REFERENCE["mask_indexed_max"], 0.01),
        ("mask-indexed r_eff mean (14 masks)", varying.mean(), REFERENCE["mask_indexed_mean"], 0.01),
        ("mask-indexed r_eff sd (14 masks)", varying.std(), REFERENCE["mask_indexed_sd"], 0.02),
        ("fixed mask 0000 r_eff", pm["0000"], REFERENCE["fixed_mask_0000"], 0.01),
        ("fixed mask 1111 r_eff", pm["1111"], REFERENCE["fixed_mask_1111"], 0.01),
        ("MC masked per-cycle mean", mc["masked_per_cycle_mean"],
         REFERENCE["mc_masked_per_cycle"], 0.01),
        ("MC masked per-cycle sd", mc["masked_per_cycle_sd"],
         REFERENCE["mc_masked_per_cycle_sd"], 0.01),
        ("MC masked sum over one cycle", mc["masked_sum_mean"],
         REFERENCE["mc_masked_sum"], 0.01),
        ("MC fixed 0000 per-cycle", mc["fixed_0000_per_cycle_mean"],
         REFERENCE["mc_fixed_0000_per_cycle"], 0.02),
        ("MC fixed 1111 per-cycle", mc["fixed_1111_per_cycle_mean"],
         REFERENCE["mc_fixed_1111_per_cycle"], 0.02),
        ("MC fixed 0000 sum", mc["fixed_0000_sum"], REFERENCE["mc_fixed_0000_sum"], 0.02),
        ("MC fixed 1111 sum", mc["fixed_1111_sum"], REFERENCE["mc_fixed_1111_sum"], 0.02),
        ("nonlinear capacity (masked)", mc["nonlinear_masked_per_cycle_mean"],
         REFERENCE["nl_masked"], 0.02),
        ("nonlinear capacity (fixed gate)", mc["nonlinear_fixed_per_cycle_mean"],
         REFERENCE["nl_fixed"], 0.02),
    ]
    print()
    print("=" * 88)
    print("SELF-CHECK against the values quoted in Supplementary Note 2")
    print("=" * 88)
    print("%-38s %13s %10s %8s" % ("quantity", "regenerated", "in SI", "status"))
    ok_all = True
    for name, got, ref, tol in checks:
        ok = abs(got - ref) <= tol
        ok_all &= ok
        print("%-38s %13.4f %10.2f %8s" % (name, got, ref, "PASS" if ok else "FAIL"))
    print("-" * 88)
    print("overall: %s" % ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return ok_all


# =========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Supplementary Note 2 analysis (relative paths only).")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="directory holding multi_states/ and matlab/ (default: ./data)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="directory for derived tables (default: ./outputs)")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the comparison against the SI values")
    args = ap.parse_args(argv)

    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("data root  : %s" % data_root)
    print("output dir : %s" % out_dir)
    print()

    dim = analyse_dimensions(data_root, out_dir)
    mc = analyse_memory_capacity(data_root, out_dir)

    results = {"dimension": dim, "memory_capacity": mc}
    with open(out_dir / "note2_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print()
    print("=" * 88)
    print("Supplementary Figure 5 uses:")
    print("  (a) outputs/note2_effrank_by_optical_input.csv")
    print("  (b) dimension.spectra_400ms in outputs/note2_results.json")
    print("  (c) outputs/note2_effrank_by_mask_word.csv")
    print("  (d) outputs/note2_memory_capacity.csv")
    print("=" * 88)

    if args.no_check:
        return 0
    return 0 if self_check(dim, mc) else 1


if __name__ == "__main__":
    sys.exit(main())
