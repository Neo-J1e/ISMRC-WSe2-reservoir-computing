#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Note 6 -- Repeatability, long-sequence stability, parameter
variation and fixed-gate control.

Regenerates the numbers quoted in Supplementary Note 6 and the panels of
Supplementary Figures 10-14:

  1. five-device repeatability        -> Supplementary Figure 10
  2. stability over 8000 steps        -> Supplementary Figure 11
  3. measured parameter variation     -> Supplementary Figures 12 and 13
  4. noise / parameter sensitivity    -> Supplementary Figure 14

No absolute path is required: all inputs are resolved relative to this file.

    python analyze_note6.py                 # everything + self-check
    python analyze_note6.py --no-figures    # tables only
    python analyze_note6.py --no-check

Outputs are written to ./outputs/.

PROVENANCE
----------
The five device endpoint voltages are read from the reviewed response traces
(``traces.npz`` plus ``c4d4/C4D4_原始响应.npz``) at the visually reviewed
response-knee index recorded in
``five_device/alignment/alignment_offsets_for_review.csv`` and
``five_device/c4d4/C4D4_时间对齐复核记录.csv``.  The original acquisition CSVs
themselves are NOT redistributed here; their provenance is recorded in the
source archives.  Every downstream quantity in Supplementary Note 6 is
reproducible from the files bundled in ``./data``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data"
DEFAULT_OUT_DIR = HERE / "outputs"

MASKS = ["%04d" % i for i in range(16)]
MASKS = [format(i, "04b") for i in range(16)]
DEVICES = ["C2D1", "C2D4", "C4D1", "C4D3", "C4D4"]
DEVICE_LABELS = {d: "Device %d" % (i + 1) for i, d in enumerate(DEVICES)}
DEVICE_CURRENT_MA = dict(zip(DEVICES, [60, 60, 40, 35, 35]))
PERM_SEED = 20260921
PERM_B = 99999
STABILITY_SESSION = "masked_20240821"     # the session used for Supplementary Fig. 11

# values quoted in Supplementary Note 6 (used by --check)
REFERENCE = {
    "sd_median_mV": [10.92, 13.92, 15.48, 5.82, 4.63],
    "sd_over_span_pct": [1.16, 1.99, 1.85, 1.26, 0.97],
    "pearson_min": 0.65, "pearson_max": 0.98, "pearson_median": 0.86,
    "friedman_q_min": 74.4, "friedman_q_max": 74.8,
    "kendall_w_min": 0.992, "kendall_w_max": 0.998,
    "holm_p": 5e-5,
    "cv_correct_total": 313, "cv_total": 400,
    "stability_shifts_mV": [1.32, 1.53, 1.73, 1.57, 2.22, 2.47, 1.59],
    "stability_final_mV": 1.59, "stability_final_sd_mV": 1.37,
    "stability_sep_min_mV": 45.7, "stability_sep_max_mV": 47.3,
    "dev4_0000_A_V": -0.17749, "dev4_0000_A_sd_mV": 2.05,
    "dev4_0000_tau_s": 0.49060, "dev4_0000_tau_sd_ms": 4.25,
    "dev4_1111_A_V": 0.29486, "dev4_1111_A_sd_mV": 2.04,
    "dev4_1111_tau_s": 0.41820, "dev4_1111_tau_sd_ms": 14.15,
    "noise_nrmse_masked_first": 0.0263, "noise_nrmse_masked_last": 0.2024,
    "noise_nrmse_fixed_first": 0.4524, "noise_nrmse_fixed_last": 0.5532,
    "noise_ser_masked_first": 0.06198, "noise_ser_masked_last": 0.07010,
    "noise_ser_fixed_first": 0.11372, "noise_ser_fixed_last": 0.18790,
}


# =========================================================================== #
# 1. five-device repeatability
# =========================================================================== #
def load_endpoint_matrices(data_root: Path):
    """(5 devices, 5 cycles, 16 masks) endpoint voltages in volts."""
    fd = data_root / "five_device"
    z = dict(np.load(fd / "traces.npz"))
    z.update(dict(np.load(fd / "c4d4" / "C4D4_原始响应.npz")))
    off = pd.concat([
        pd.read_csv(fd / "alignment" / "alignment_offsets_for_review.csv",
                    dtype={"mask": str}),
        pd.read_csv(fd / "c4d4" / "C4D4_时间对齐复核记录.csv", dtype={"mask": str}),
    ]).set_index(["device", "cycle", "mask"])

    xs = {}
    for dev in DEVICES:
        xs[dev] = np.array([
            [z["%s_%d_%s" % (dev, n, m)][int(off.loc[(dev, n, m), "applied_off_index"])]
             for m in MASKS] for n in range(1, 6)])
    return xs


def analyse_five_devices(data_root: Path, out_dir: Path, make_figures: bool) -> dict:
    print("=" * 96)
    print("SECTION 1  |  five-device repeatability (Supplementary Figure 10)")
    print("=" * 96)
    xs = load_endpoint_matrices(data_root)
    pairs = list(itertools.combinations(range(16), 2))
    rng = np.random.default_rng(PERM_SEED)      # one stream, consumed device by device

    rows, pair_rows, cv_rows = [], [], []
    print("%-8s %9s %10s %8s %9s %9s %8s %8s %7s %7s"
          % ("device", "span_mV", "SDmed_mV", "SD/span", "Q", "W", "p", "holm_p",
             "overlap", "CV"))
    for dev in DEVICES:
        x = xs[dev]
        n, k = x.shape
        mu, sd = x.mean(0), x.std(0, ddof=1)
        span = float(np.ptp(mu))

        ranks = np.argsort(np.argsort(x, axis=1), axis=1) + 1
        q = 12 / (n * k * (k + 1)) * np.sum(ranks.sum(0) ** 2) - 3 * n * (k + 1)
        count = 0
        for b in range(0, PERM_B, 1000):
            size = min(1000, PERM_B - b)
            rr = rng.permuted(np.broadcast_to(ranks, (size, n, k)), axis=2)
            qq = 12 / (n * k * (k + 1)) * np.sum(rr.sum(1) ** 2, axis=1) - 3 * n * (k + 1)
            count += int((qq >= q - 1e-9).sum())
        p = (count + 1) / (PERM_B + 1)

        pred = np.array([abs(x[h, :, None] - x[np.arange(n) != h].mean(0)[None, :]).argmin(1)
                         for h in range(n)])
        correct = int((pred == np.arange(16)[None, :]).sum())
        overlap = int(sum(max(x[:, i].min(), x[:, j].min())
                          <= min(x[:, i].max(), x[:, j].max()) for i, j in pairs))

        rows.append({
            "device": DEVICE_LABELS[dev], "internal_id": dev,
            "laser_current_mA": DEVICE_CURRENT_MA[dev],
            "span_mV": span * 1000,
            "sd_median_mV": float(np.median(sd) * 1000),
            "sd_over_span_pct": float(np.median(sd) / span * 100),
            "pooled_sd_mV": float(np.sqrt(np.mean(sd ** 2)) * 1000),
            "friedman_q": float(q), "kendall_w": float(q / (n * (k - 1))),
            "permutation_p": float(p), "overlap_pairs": overlap,
            "cv_correct": correct, "cv_total": n * 16,
        })
        for i, j in pairs:
            pair_rows.append({
                "device": DEVICE_LABELS[dev], "internal_id": dev,
                "mask_a": MASKS[i], "mask_b": MASKS[j],
                "mean_gap_mV": float(abs(mu[i] - mu[j]) * 1000),
                "observed_range_overlap": bool(
                    max(x[:, i].min(), x[:, j].min()) <= min(x[:, i].max(), x[:, j].max())),
                "standardized_gap": float(abs(mu[i] - mu[j]) / np.sqrt(sd[i] ** 2 + sd[j] ** 2)),
            })
        for h in range(n):
            for j, m in enumerate(MASKS):
                cv_rows.append({"device": DEVICE_LABELS[dev], "internal_id": dev,
                                "cycle": h + 1, "mask": m,
                                "predicted_mask": MASKS[pred[h, j]],
                                "correct": bool(m == MASKS[pred[h, j]])})

    s = pd.DataFrame(rows)
    order = np.argsort(s.permutation_p.values)
    adj = np.maximum.accumulate(s.permutation_p.values[order] * (5 - np.arange(5)))
    s["holm_p"] = 0.0
    s.loc[order, "holm_p"] = np.minimum(adj, 1)
    for _, r in s.iterrows():
        print("%-8s %9.2f %10.2f %7.2f%% %9.3f %9.5f %8.5f %8.5f %7d %3d/%d"
              % (r.device, r.span_mV, r.sd_median_mV, r.sd_over_span_pct, r.friedman_q,
                 r.kendall_w, r.permutation_p, r.holm_p, r.overlap_pairs,
                 r.cv_correct, r.cv_total))

    means = np.array([xs[d].mean(0) for d in DEVICES])
    pearson = [float(np.corrcoef(means[i], means[j])[0, 1])
               for i, j in itertools.combinations(range(len(DEVICES)), 2)]
    print("-" * 96)
    print("pairwise Pearson between the five mean mask-response profiles: "
          "min %.2f  median %.2f  max %.2f"
          % (min(pearson), float(np.median(pearson)), max(pearson)))
    print("leave-one-cycle-out mask identification: %d/%d = %.2f%%"
          % (s.cv_correct.sum(), s.cv_total.sum(), s.cv_correct.sum() / s.cv_total.sum() * 100))

    s.to_csv(out_dir / "note6_five_device_summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out_dir / "note6_pairwise_overlap.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(out_dir / "note6_cv_predictions.csv", index=False)
    pd.DataFrame(means.T * 1000, index=MASKS,
                 columns=[DEVICE_LABELS[d] for d in DEVICES]
                 ).to_csv(out_dir / "note6_mean_endpoint_by_mask.csv")

    if make_figures:
        _figure_s10(xs, means, out_dir)

    return {
        "devices": rows,
        "pearson": {"min": min(pearson), "median": float(np.median(pearson)),
                    "max": max(pearson)},
        "cv_correct_total": int(s.cv_correct.sum()),
        "cv_total": int(s.cv_total.sum()),
        "holm_p": [float(v) for v in s.holm_p],
    }


def _figure_s10(xs, means, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#0070C0", "#D85E00", "#188267", "#A04DA2", "#636363"]
    xx = np.arange(16)
    fig, axs = plt.subplots(3, 2, figsize=(7.2, 8.4), sharey=True)
    for dev, ax in zip(DEVICES, axs.flat):
        y = xs[dev] * 1000
        ax.errorbar(xx, y.mean(0), yerr=y.std(0, ddof=1), fmt="_", color="black",
                    markersize=8, capsize=2.5, lw=0.9, label="Mean $\\pm$ sample SD", zorder=2)
        for n, c in enumerate(colors):
            ax.scatter(xx + (n - 2) * 0.10, y[n], color=c, s=11, label="Cycle %d" % (n + 1),
                       zorder=3)
        ax.set_title("%s   laser drive %d mA" % (DEVICE_LABELS[dev], DEVICE_CURRENT_MA[dev]),
                     loc="left", fontsize=9, fontweight="bold")
        ax.set_xticks(xx)
        ax.set_xticklabels(MASKS, rotation=90, fontsize=7)
        ax.grid(axis="y", alpha=0.15)
        ax.set_ylabel("Endpoint $V_{OC}$ (mV)")
        ax.set_xlabel("Mask word")
    ax = axs.flat[5]
    ax.axis("off")
    palette = ["#0070C0", "#188267", "#D85E00", "#A04DA2", "#9C3636"]
    for dev, y, c in zip(DEVICES, means, palette):
        ax.plot(xx, y * 1000, "o-", lw=0.8, ms=2.8, label=DEVICE_LABELS[dev], color=c)
    ax.set_xticks(xx)
    ax.set_xticklabels(MASKS, rotation=90, fontsize=7)
    ax.set_ylabel("Mean endpoint $V_{OC}$ (mV)")
    ax.set_xlabel("Mask word")
    ax.set_title("(f) mean mask-response profiles", loc="left", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.legend(*axs.flat[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.945, bottom=0.07, left=0.10, right=0.99, hspace=0.62, wspace=0.12)
    fig.savefig(out_dir / "figS10_device_repeatability.png", dpi=400, facecolor="white")
    plt.close(fig)


# =========================================================================== #
# 2. stability over 8000 steps
# =========================================================================== #
def analyse_stability(data_root: Path, out_dir: Path, make_figures: bool) -> dict:
    print()
    print("=" * 96)
    print("SECTION 2  |  stability over 8000 measured time steps (Supplementary Figure 11)")
    print("=" * 96)
    blocks = json.loads((data_root / "stability" / "stability_8000steps.json")
                        .read_text(encoding="utf-8"))
    blk = blocks[STABILITY_SESSION]
    seg = np.array([p["seg_mean_mV"] for p in blk["per_config"]])       # (14, 8)
    shifts = np.abs(seg[:, 1:] - seg[:, [0]])
    sep = np.array(blk["pairwise_separation_mV"]) * 1000                # stored in V
    shift_mean = shifts.mean(0)
    print("session            : %s   (%d masks, %d steps, %d blocks of %d)"
          % (STABILITY_SESSION, blk["n_configurations"], blk["n_steps"],
             len(blk["segments"]), blk["seg_steps"]))
    print("mean |block shift| relative to block 1, blocks 2-8 (mV): %s"
          % [round(v, 2) for v in shift_mean])
    print("final block shift  : %.2f +/- %.2f mV across the %d masks"
          % (shift_mean[-1], shifts[:, -1].std(ddof=1), blk["n_configurations"]))
    print("mean pairwise mask separation (mV): %s" % [round(v, 1) for v in sep])
    print("separation range   : %.1f - %.1f mV" % (sep.min(), sep.max()))

    pd.DataFrame({
        "block": np.arange(1, 9),
        # block 1 is the reference, so its shift is zero by definition
        "mean_abs_shift_mV": np.concatenate([[0.0], shift_mean]),
        "sd_abs_shift_mV": np.concatenate([[0.0], shifts.std(0, ddof=1)]),
        "mean_pairwise_separation_mV": sep,
    }).to_csv(out_dir / "note6_stability_blocks.csv", index=False)
    pd.DataFrame(seg, index=[p["idx"] for p in blk["per_config"]],
                 columns=["block_%d" % (i + 1) for i in range(seg.shape[1])]
                 ).to_csv(out_dir / "note6_stability_segment_means.csv")

    if make_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        bx = np.arange(1, 9)
        for row in seg:
            ax.plot(bx, row - row[0], color="#9a9a9a", lw=0.8, alpha=0.9, zorder=1)
        ax.errorbar(bx[1:], shift_mean, yerr=shifts.std(0, ddof=1), fmt="o-", color="black",
                    ms=4, lw=1.2, capsize=2.5, zorder=3,
                    label="mean $\\pm$ sample SD across 14 masks")
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel("Block index (1000 consecutive time steps each)")
        ax.set_ylabel("Block-mean $V_{OC}$ change (mV)")
        ax.set_xticks(bx)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
        fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.16)
        fig.savefig(out_dir / "figS11_stability_8000steps.png", dpi=400, facecolor="white")
        plt.close(fig)

    return {
        "session": STABILITY_SESSION,
        "shifts_mV": shift_mean.tolist(),
        "final_shift_mV": float(shift_mean[-1]),
        "final_shift_sd_mV": float(shifts[:, -1].std(ddof=1)),
        "separation_min_mV": float(sep.min()),
        "separation_max_mV": float(sep.max()),
        "mean_pairwise_separation_mV": float(blk["summary"]["mean_pairwise_separation_mV"]),
    }


# =========================================================================== #
# 3. measured parameter variation (Device 4)
# =========================================================================== #
def analyse_parameter_variation(data_root: Path, out_dir: Path, make_figures: bool) -> dict:
    print()
    print("=" * 96)
    print("SECTION 3  |  measured cycle-to-cycle parameter variation "
          "(Supplementary Figures 12 and 13)")
    print("=" * 96)
    pv = data_root / "parameter_variation"
    fit = pd.read_csv(pv / "Device4_逐cycle拟合参数.csv")
    traces = pd.read_csv(pv / "Device4_10条原始与拟合数据.csv")

    out = {}
    print("%-8s %12s %10s %12s %10s" % ("program", "A_mean_V", "A_sd_mV", "tau_mean_s", "tau_sd_ms"))
    for mask, label in ((0, "0000"), (1111, "1111")):
        g = fit[fit["mask"] == mask]
        A, tau = g["A_fit_V"].to_numpy(), g["tau_fit_s"].to_numpy()
        key = "dev4_%s" % label
        out[key] = {"A_mean_V": float(A.mean()), "A_sd_mV": float(A.std(ddof=1) * 1000),
                    "tau_mean_s": float(tau.mean()),
                    "tau_sd_ms": float(tau.std(ddof=1) * 1000),
                    "n_cycles": int(len(g))}
        print("%-8s %12.5f %10.2f %12.5f %10.2f"
              % (label, A.mean(), A.std(ddof=1) * 1000, tau.mean(), tau.std(ddof=1) * 1000))

    fit.to_csv(out_dir / "note6_device4_cycle_parameters.csv", index=False)

    if make_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["#0070C0", "#D85E00", "#188267", "#A04DA2", "#636363"]
        fig, axs = plt.subplots(2, 5, figsize=(9.0, 4.4), sharex=True)
        for row, (mask, label) in enumerate(((0, "0000"), (1111, "1111"))):
            sub = traces[traces["mask"] == mask]
            ymin = sub["measured_V"].min()
            ymax = sub["measured_V"].max()
            for col, cyc in enumerate(sorted(sub["cycle"].unique())):
                ax = axs[row, col]
                t = sub[sub["cycle"] == cyc]
                ax.plot(t["time_from_light_on_s"], t["measured_V"], color=colors[col],
                        lw=1.1, label="measured")
                ax.plot(t["time_from_light_on_s"], t["single_exponential_fit_V"], "r--",
                        lw=0.9, label="fit")
                g = fit[(fit["mask"] == mask) & (fit["cycle"] == cyc)].iloc[0]
                ax.set_title("cycle %d\nA=%.4f V, $\\tau$=%.3f s\n$R^2$=%.4f"
                             % (cyc, g["A_fit_V"], g["tau_fit_s"], g["fit_R2"]), fontsize=6.5)
                ax.set_ylim(ymin - 0.02, ymax + 0.02)
                ax.tick_params(labelsize=6)
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel("$V_{OC}$ (V), program %s" % label, fontsize=7.5)
                if row == 1:
                    ax.set_xlabel("Time after light on (s)", fontsize=7)
        axs[0, 0].legend(frameon=False, fontsize=6.5, loc="lower right")
        fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.10,
                            hspace=0.42, wspace=0.28)
        fig.savefig(out_dir / "figS12_13_device4_calibration.png", dpi=400, facecolor="white")
        plt.close(fig)

    return out


# =========================================================================== #
# 4. noise / parameter sensitivity
# =========================================================================== #
def analyse_noise(data_root: Path, out_dir: Path, make_figures: bool) -> dict:
    print()
    print("=" * 96)
    print("SECTION 4  |  sensitivity to absolute parameter perturbations "
          "(Supplementary Figure 14)")
    print("=" * 96)
    ns = data_root / "noise_sweep"
    specs = [("classification", "NRMSE"), ("equalization", "SER")]
    out = {}
    print("%-16s %-8s %10s %10s %8s" % ("task", "config", "first", "last", "n_real"))
    for task, metric in specs:
        raw = pd.read_csv(ns / ("fig_noise_sweep_CI_%s_raw.csv" % task))
        for cfg in ("masked", "fixedgate"):
            g = raw[raw["configuration"] == cfg]
            by = g.groupby("sigma")[metric].mean()
            out["%s_%s" % (task, cfg)] = {
                "first_sigma": float(by.index[0]), "first": float(by.iloc[0]),
                "last_sigma": float(by.index[-1]), "last": float(by.iloc[-1]),
                "n_realizations": int(g["realization"].nunique()),
            }
            print("%-16s %-8s %10.5f %10.5f %8d"
                  % (task, cfg, by.iloc[0], by.iloc[-1], g["realization"].nunique()))
        summary = pd.read_csv(ns / ("fig_noise_sweep_CI_%s.csv" % task))
        summary.to_csv(out_dir / ("note6_noise_%s_summary.csv" % task), index=False)

    if make_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1))
        for ax, (task, metric), ylab in zip(
                axs, specs, ["NRMSE", "Symbol error rate (SER)"]):
            summary = pd.read_csv(ns / ("fig_noise_sweep_CI_%s.csv" % task))
            for cfg, color, lab in (("masked", "#C0392B", "with masking"),
                                    ("fixedgate", "#2E6FBB", "fixed gate")):
                g = summary[summary["configuration"] == cfg].sort_values("sigma")
                ax.errorbar(g["sigma"], g["mean_" + metric],
                            yerr=[g["mean_" + metric] - g["ci95_low"],
                                  g["ci95_high"] - g["mean_" + metric]],
                            fmt="o-", ms=3.6, lw=1.1, capsize=2, color=color, label=lab)
            ax.set_xlabel("Perturbation $\\sigma$ (V for $A$, s for $\\tau$)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, fontsize=8)
        fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.20, wspace=0.34)
        fig.savefig(out_dir / "figS14_noise_sensitivity.png", dpi=400, facecolor="white")
        plt.close(fig)
    return out


# =========================================================================== #
# self-check
# =========================================================================== #
def self_check(fd: dict, st: dict, pv: dict, nz: dict) -> bool:
    checks = []
    for i, row in enumerate(fd["devices"]):
        checks.append(("median cycle-to-cycle SD, %s (mV)" % row["device"],
                       row["sd_median_mV"], REFERENCE["sd_median_mV"][i], 0.01))
    for i, row in enumerate(fd["devices"]):
        checks.append(("%s SD / span (%%)" % row["device"],
                       row["sd_over_span_pct"], REFERENCE["sd_over_span_pct"][i], 0.01))
    checks += [
        ("pairwise Pearson, min", fd["pearson"]["min"], REFERENCE["pearson_min"], 0.01),
        ("pairwise Pearson, max", fd["pearson"]["max"], REFERENCE["pearson_max"], 0.01),
        ("pairwise Pearson, median", fd["pearson"]["median"], REFERENCE["pearson_median"], 0.01),
        ("Friedman Q, min", min(r["friedman_q"] for r in fd["devices"]),
         REFERENCE["friedman_q_min"], 0.05),
        ("Friedman Q, max", max(r["friedman_q"] for r in fd["devices"]),
         REFERENCE["friedman_q_max"], 0.05),
        ("Kendall W, min", min(r["kendall_w"] for r in fd["devices"]),
         REFERENCE["kendall_w_min"], 0.001),
        ("Kendall W, max", max(r["kendall_w"] for r in fd["devices"]),
         REFERENCE["kendall_w_max"], 0.001),
        ("Holm-adjusted p", max(fd["holm_p"]), REFERENCE["holm_p"], 1e-9),
        ("leave-one-cycle-out correct", fd["cv_correct_total"],
         REFERENCE["cv_correct_total"], 0),
        ("final block shift (mV)", st["final_shift_mV"], REFERENCE["stability_final_mV"], 0.01),
        ("final block shift SD (mV)", st["final_shift_sd_mV"],
         REFERENCE["stability_final_sd_mV"], 0.01),
        ("pairwise separation, min (mV)", st["separation_min_mV"],
         REFERENCE["stability_sep_min_mV"], 0.1),
        ("pairwise separation, max (mV)", st["separation_max_mV"],
         REFERENCE["stability_sep_max_mV"], 0.1),
        ("Device 4 / 0000 target A (V)", pv["dev4_0000"]["A_mean_V"],
         REFERENCE["dev4_0000_A_V"], 1e-5),
        ("Device 4 / 0000 A SD (mV)", pv["dev4_0000"]["A_sd_mV"],
         REFERENCE["dev4_0000_A_sd_mV"], 0.01),
        ("Device 4 / 0000 tau (s)", pv["dev4_0000"]["tau_mean_s"],
         REFERENCE["dev4_0000_tau_s"], 1e-5),
        ("Device 4 / 0000 tau SD (ms)", pv["dev4_0000"]["tau_sd_ms"],
         REFERENCE["dev4_0000_tau_sd_ms"], 0.01),
        ("Device 4 / 1111 target A (V)", pv["dev4_1111"]["A_mean_V"],
         REFERENCE["dev4_1111_A_V"], 1e-5),
        ("Device 4 / 1111 A SD (mV)", pv["dev4_1111"]["A_sd_mV"],
         REFERENCE["dev4_1111_A_sd_mV"], 0.01),
        ("Device 4 / 1111 tau (s)", pv["dev4_1111"]["tau_mean_s"],
         REFERENCE["dev4_1111_tau_s"], 1e-5),
        ("Device 4 / 1111 tau SD (ms)", pv["dev4_1111"]["tau_sd_ms"],
         REFERENCE["dev4_1111_tau_sd_ms"], 0.01),
        ("NRMSE masked, smallest sigma", nz["classification_masked"]["first"],
         REFERENCE["noise_nrmse_masked_first"], 1e-4),
        ("NRMSE masked, largest sigma", nz["classification_masked"]["last"],
         REFERENCE["noise_nrmse_masked_last"], 1e-4),
        ("NRMSE fixed gate, smallest sigma", nz["classification_fixedgate"]["first"],
         REFERENCE["noise_nrmse_fixed_first"], 1e-4),
        ("NRMSE fixed gate, largest sigma", nz["classification_fixedgate"]["last"],
         REFERENCE["noise_nrmse_fixed_last"], 1e-4),
        ("SER masked, smallest sigma", nz["equalization_masked"]["first"],
         REFERENCE["noise_ser_masked_first"], 1e-5),
        ("SER masked, largest sigma", nz["equalization_masked"]["last"],
         REFERENCE["noise_ser_masked_last"], 1e-5),
        ("SER fixed gate, smallest sigma", nz["equalization_fixedgate"]["first"],
         REFERENCE["noise_ser_fixed_first"], 1e-5),
        ("SER fixed gate, largest sigma", nz["equalization_fixedgate"]["last"],
         REFERENCE["noise_ser_fixed_last"], 1e-5),
    ]
    for i, ref in enumerate(REFERENCE["stability_shifts_mV"]):
        checks.append(("block %d mean abs shift (mV)" % (i + 2),
                       st["shifts_mV"][i], ref, 0.01))

    print()
    print("=" * 96)
    print("SELF-CHECK against the values quoted in Supplementary Note 6")
    print("=" * 96)
    print("%-44s %14s %12s %8s" % ("quantity", "regenerated", "in SI", "status"))
    ok_all = True
    for name, got, ref, tol in checks:
        ok = abs(got - ref) <= tol
        ok_all &= ok
        print("%-44s %14.5f %12.5f %8s" % (name, got, ref, "PASS" if ok else "FAIL"))
    print("-" * 96)
    print("overall: %s" % ("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED"))
    return ok_all


# =========================================================================== #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Supplementary Note 6 analysis (relative paths only).")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args(argv)

    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = not args.no_figures
    print("data root  : %s" % data_root)
    print("output dir : %s" % out_dir)
    print("figures    : %s" % ("yes" if figs else "no"))
    print()

    fd = analyse_five_devices(data_root, out_dir, figs)
    st = analyse_stability(data_root, out_dir, figs)
    pv = analyse_parameter_variation(data_root, out_dir, figs)
    nz = analyse_noise(data_root, out_dir, figs)

    with open(out_dir / "note6_results.json", "w", encoding="utf-8") as fh:
        json.dump({"five_device": fd, "stability": st,
                   "parameter_variation": pv, "noise": nz}, fh, indent=2)
    print()
    print("Supplementary Figures 10-14 regenerated in %s" % out_dir)

    if args.no_check:
        return 0
    return 0 if self_check(fd, st, pv, nz) else 1


if __name__ == "__main__":
    sys.exit(main())
