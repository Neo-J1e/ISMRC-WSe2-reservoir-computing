# -*- coding: utf-8 -*-
"""Draw Figure R7 of the response to referees: all 14 mask sequences.

Same content as Supplementary Fig. 9, in the landscape layout used for the
response letter. Reads the residuals written by run_reproducibility_audit.m.

    python plot_R7_14_mask_validation.py  ->  outputs/Figure_R7_14masks.png
"""
import io
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "outputs", "classification_continuous_validation_residuals.csv")
MET = os.path.join(HERE, "outputs", "classification_continuous_validation_metrics.csv")
LASER = os.path.join(HERE, "data", "wave_classification_laser.txt")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9.5,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.unicode_minus": False,
    "axes.labelsize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.minor.size": 1.4, "ytick.minor.size": 1.4,
    "xtick.top": True, "ytick.right": True,
    "axes.grid": False,
})

per = defaultdict(dict)
with io.open(RES, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        per[r["mask_word"]][int(r["sample_index"])] = (float(r["measured_V"]),
                                                       float(r["predicted_V"]))
metric = {}
with io.open(MET, encoding="utf-8-sig") as f:
    for m in csv.DictReader(f):
        metric[m["mask_word"]] = float(m["pearson_r"])
masks = sorted(per, key=lambda m: -metric.get(m, 0))

laser = np.loadtxt(LASER)
N = 300
BLUE, RED, GREEN = "#3B6FD4", "#D93A3A", "#A8C881"
IV = 37.0 + 6.0 * laser[:75]

# ---------------------------------------------------------------- geometry (inches)
FIG_W, FIG_H = 6.9, 7.9
L, R = 0.42, 6.86
GAPX = 0.055
PER_ROW = 7
COLW = (R - L - (PER_ROW - 1) * GAPX) / PER_ROW

A_TOP, A_H = 0.30, 0.52
A_XLAB = 0.17
ROW_TOP0 = 1.30
ROW_H = 1.10                      # taller mask panels
TICK_DY = 0.16                    # tick numbers below the frame
TS_DY = 0.17                      # 'Time step' just under the tick numbers
WORD_DY = 0.17                    # mask word line above the frame
LETTER_DY = 0.17                  # letter line, above the mask word
ROW_STRIDE = ROW_H + TICK_DY + TS_DY + 0.16 + LETTER_DY + WORD_DY


def ax_rect(x_in, y_top_in, w_in, h_in):
    return [x_in / FIG_W, 1.0 - (y_top_in + h_in) / FIG_H, w_in / FIG_W, h_in / FIG_H]


fig = plt.figure(figsize=(FIG_W, FIG_H))

# ---------------- (a): tick numbers at the bottom
axa = fig.add_axes(ax_rect(L, A_TOP, R - L, A_H))
axa.step(np.arange(1, 76) * 4 - 1.5, IV, where="mid", color=GREEN, lw=1.6)
axa.set_xlim(0, N); axa.set_ylim(15, 150)
axa.set_yticks([40, 80, 120])
axa.set_xticks([0, 100, 200, 300])
axa.tick_params(axis="x", labelbottom=True, labeltop=False, pad=2)
axa.set_ylabel(r"$I_\mathrm{laser}$ (mA)", fontsize=10, labelpad=1)
axa.text(-0.028, 1.03, "a", transform=axa.transAxes, fontsize=16, fontweight="bold",
         va="bottom", ha="left")

letters = "bcdefghijklmno"
for ri in range(2):
    y_top = ROW_TOP0 + ri * ROW_STRIDE
    for ci in range(PER_ROW):
        k = ri * PER_ROW + ci
        if k >= len(masks):
            break
        m = masks[k]
        idx = sorted(per[m])[:N]
        meas = np.array([per[m][i][0] for i in idx])
        pred = np.array([per[m][i][1] for i in idx])
        x0 = L + ci * (COLW + GAPX)
        ax1 = fig.add_axes(ax_rect(x0, y_top, COLW, ROW_H / 2))
        ax2 = fig.add_axes(ax_rect(x0, y_top + ROW_H / 2, COLW, ROW_H / 2),
                           sharex=ax1, sharey=ax1)
        ax1.plot(np.arange(1, N + 1), pred, color=BLUE, lw=0.35)
        ax2.plot(np.arange(1, N + 1), meas, color=RED, lw=0.35)
        ax1.set_xlim(1, N); ax1.set_ylim(-0.8, 0.6)
        ax1.set_yticks([-0.5, 0, 0.5])
        ax1.set_xticks([]); ax2.set_xticks([])
        ax1.tick_params(labelbottom=False)
        if ci == 0:
            ax1.set_ylabel(r"$V_\mathrm{OC}$ (V)", fontsize=9.5, labelpad=1)
            ax2.set_ylabel(r"$V_\mathrm{OC}$ (V)", fontsize=9.5, labelpad=1)
        else:
            ax1.tick_params(labelleft=False); ax2.tick_params(labelleft=False)
        ax1.text(0.05, 0.05, "i)", transform=ax1.transAxes, fontsize=7.5,
                 va="bottom", ha="left")
        ax2.text(0.05, 0.05, "ii)", transform=ax2.transAxes, fontsize=7.5,
                 va="bottom", ha="left")
        # manual tick numbers: '0' at the left edge, '300' pulled inwards
        ynum = 1.0 - (y_top + ROW_H + TICK_DY) / FIG_H
        fig.text(x0 / FIG_W, ynum, "0", fontsize=7.5, va="baseline", ha="left")
        fig.text((x0 + COLW - 0.080) / FIG_W, ynum, "300",
                 fontsize=7.5, va="baseline", ha="right")
        # mask word centred, on the line above; letter at the panel's top-left
        fig.text((x0 + COLW / 2) / FIG_W, 1.0 - (y_top - WORD_DY) / FIG_H, m,
                 fontsize=9.5, va="baseline", ha="center")
        fig.text((x0 - 0.010) / FIG_W, 1.0 - (y_top - LETTER_DY) / FIG_H, letters[k],
                 fontsize=15, fontweight="bold", va="baseline", ha="left")
    fig.text((L + (R - L) / 2) / FIG_W,
             1.0 - (y_top + ROW_H + TICK_DY + TS_DY) / FIG_H,
             "Time step", fontsize=9.5, va="baseline", ha="center")

OUT = os.path.join(HERE, "outputs", "Figure_R7_14masks.png")
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)
print(OUT)
