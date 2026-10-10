"""Draw Supplementary Fig. 9: the measured/model comparison for all 14 mask sequences.

Reads the residuals produced by run_reproducibility_audit.m and the input waveform.
Source voltage traces are used without normalization, shifting or smoothing.

    python plot_14_mask_validation.py   ->  outputs/classification_continuous_validation_14masks.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESIDUALS = HERE / "outputs" / "classification_continuous_validation_residuals.csv"
LASER_INPUT = HERE / "data" / "wave_classification_laser.txt"
OUTPUT = HERE / "outputs" / "classification_continuous_validation_14masks.png"

MASK_ROWS = [
    ["0101", "0011", "1001", "0001", "0010", "1110", "1010"],
    ["0110", "1101", "1011", "1100", "0100", "0111", "1000"],
]
PANEL_ROWS = [list("bcdefgh"), list("ijklmno")]
WINDOW = 300


def read_traces():
    traces = {mask: {"step": [], "measured": [], "model": []}
              for row in MASK_ROWS for mask in row}
    with RESIDUALS.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            mask = record["mask_word"]
            if mask not in traces:
                continue
            traces[mask]["step"].append(int(record["sample_index"]))
            traces[mask]["measured"].append(float(record["measured_V"]))
            traces[mask]["model"].append(float(record["predicted_V"]))
    for mask, trace in traces.items():
        if trace["step"] != list(range(1, 8001)):
            raise ValueError(f"Unexpected sample indices for mask {mask}")
    return traces


def style_axis(ax):
    ax.tick_params(axis="both", which="major", direction="in", top=True,
                   right=True, length=2.5, width=0.7, labelsize=7.5, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

traces = read_traces()
normalized_input = np.loadtxt(LASER_INPUT, dtype=float).reshape(-1)
laser_current = np.repeat(37.0 + 6.0 * normalized_input, 4)[:WINDOW]

fig = plt.figure(figsize=(6.9, 6.9), dpi=300)

# Fixed geometry gives all seven narrow columns identical widths and makes the
# two axes within each mask panel meet exactly at their shared boundary.
left, right = 0.092, 0.987
column_gap = 0.014
column_width = (right - left - 6 * column_gap) / 7
pair_positions = [(0.470, 0.770), (0.065, 0.365)]
half_height = 0.150

# Panel (a): a full-width input strip.
ax_a = fig.add_axes([left, 0.845, right - left, 0.120])
edges = np.arange(WINDOW + 1)
ax_a.stairs(laser_current, edges, color="#A8C881", linewidth=1.6)
ax_a.set_xlim(0, 300)
ax_a.set_ylim(15, 150)
ax_a.set_yticks([40, 80, 120])
ax_a.set_ylabel(r"$I_{\mathrm{laser}}$ (mA)")
ax_a.set_xticks([0, 100, 200, 300])
ax_a.tick_params(axis="x", labeltop=False, labelbottom=True, top=True,
                 bottom=True, direction="in", labelsize=7.5, pad=2)
ax_a.get_xticklabels()[0].set_ha("left")
ax_a.get_xticklabels()[-1].set_ha("right")
ax_a.tick_params(axis="y", left=True, right=True, direction="in",
                 labelsize=7.5, pad=2)
for spine in ax_a.spines.values():
    spine.set_linewidth(0.7)
ax_a.text(-0.035, 1.035, "a", transform=ax_a.transAxes, fontsize=15,
          fontweight="bold", ha="left", va="bottom", clip_on=False)

for row_index, (masks, letters) in enumerate(zip(MASK_ROWS, PANEL_ROWS)):
    pair_bottom, pair_top = pair_positions[row_index]
    for column_index, (mask, letter) in enumerate(zip(masks, letters)):
        x0 = left + column_index * (column_width + column_gap)
        ax_bottom = fig.add_axes([x0, pair_bottom, column_width, half_height])
        ax_top = fig.add_axes([x0, pair_bottom + half_height,
                               column_width, half_height], sharex=ax_bottom,
                              sharey=ax_bottom)
        trace = traces[mask]
        steps = np.asarray(trace["step"][:WINDOW])
        ax_top.plot(steps, trace["model"][:WINDOW], color="#3B6FD4", lw=0.35)
        ax_bottom.plot(steps, trace["measured"][:WINDOW], color="#D93A3A", lw=0.35)

        for ax in (ax_top, ax_bottom):
            style_axis(ax)
            ax.set_xlim(0, 300)
            ax.set_ylim(-0.8, 0.6)
            ax.set_yticks([-0.5, 0, 0.5])
            ax.set_xticks([0, 300])

        ax_top.tick_params(labelbottom=False)
        ax_bottom.set_xlabel("Time step", fontsize=9.5, labelpad=3)
        # Keep the two endpoint labels inside each narrow panel so adjacent
        # columns retain visibly separate "300" and "0" labels.
        ax_bottom.get_xticklabels()[0].set_ha("left")
        ax_bottom.get_xticklabels()[-1].set_ha("right")
        if column_index == 0:
            ax_bottom.set_ylabel(r"$V_{\mathrm{OC}}$ (V)", fontsize=9.5,
                                 labelpad=9)
            # Centre the single row label across the touching pair.
            ax_bottom.yaxis.set_label_coords(-0.43, 1.0)
        else:
            ax_top.tick_params(labelleft=False)
            ax_bottom.tick_params(labelleft=False)

        # Letter and mask share a baseline outside the upper-left frame.
        header_y = pair_top + 0.012
        fig.text(x0 - 0.010, header_y, letter, fontsize=15, fontweight="bold",
                 ha="left", va="bottom")
        fig.text(x0 + column_width / 2, header_y + 0.002, mask, fontsize=9.5,
                 ha="center", va="bottom")

fig.savefig(OUTPUT, dpi=300, facecolor="white", edgecolor="none")
plt.close(fig)
print(OUTPUT)
