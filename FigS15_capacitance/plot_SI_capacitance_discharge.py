# -*- coding: utf-8 -*-
"""Supplementary Figure 15: node-capacitance dependence of the light response.

Figure conventions:
    * Arial throughout, mathtext included; font.size 17, axis labels 19,
      tick labels 13.5/15, legend 14
    * axes frame linewidth 2.0; ticks inward on all four sides; no grid
    * no title inside a panel, the caption carries the information; panel
      letters 'a'/'b'/'c' at (-0.24, 1.02), fontsize 30, bold
    * panels share width and height (ax.set_box_aspect(1))
    * tab:blue / tab:red for the two main groups, tab:orange for the third
    * white background, dpi 300
    * no concluding sentence on the figure

Usage::

    python plot_SI_capacitance_discharge.py            # 100nF / 10nF / 1nF x mask 1111/0000
    python plot_SI_capacitance_discharge.py --no-tau   # legend without tau

Output: figures/SI/FigS_capacitance_discharge.{png,svg,pdf} + _caption.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

# ---- figure conventions, applied verbatim --------------------------------- #
plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 17,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
    'axes.unicode_minus': False,
    'axes.labelsize': 19, 'xtick.labelsize': 13.5, 'ytick.labelsize': 15,
    'legend.fontsize': 14,
    'axes.linewidth': 2.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'xtick.minor.width': 0.9, 'ytick.minor.width': 0.9,
    'xtick.major.size': 4.0, 'ytick.major.size': 4.0,
    'xtick.minor.size': 2.2, 'ytick.minor.size': 2.2,
    'xtick.top': True, 'ytick.right': True,
    'axes.grid': False,
})
plt.rcParams['legend.frameon'] = False          # frameless legend, as in the other figures
plt.rcParams['savefig.facecolor'] = 'white'

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_manifest import load_run, load_runs                         # noqa: E402

BASE = os.path.join(HERE, "mask_test_data")
OUTDIR = os.path.join(HERE, "figures", "SI")
STEM = "FigS_capacitance_discharge"

CAP_UNITS = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3}
CAP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(p|n|u|µ|m)F", re.I)

#: colour encodes the node capacitance (blue primary, red control, orange third)
#: keyed by integer pF: 100 * 1e-9 is not exactly 100e-9, a float key would raise KeyError
CAP_COLOR = {100000: "tab:blue", 10000: "tab:red", 1000: "tab:orange"}
#: line style encodes the gate mask configuration
MASK_STYLE = {"1111": "-", "0000": "--"}


def pF(c):
    """Integer pF key of a capacitance."""
    return int(round(float(c) * 1e12))


def cap_label(c):
    if c >= 1e-9:
        return f"{c / 1e-9:g} nF"
    return f"{c / 1e-12:g} pF"


def load(d, last=None):
    """One run, from runs.csv plus its response trace."""
    return load_run(d)


def tau_of(r):
    v, dt, i0 = r["v"], r["dt"], r["n_settle"]
    i1 = min(i0 + r["active"], v.size - 1)
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
        return dict(v_dark=v_dark, step=step, tau=float("nan"), r2=float("nan"))
    a, b = np.polyfit(t[idx], np.log(y[idx]), 1)
    pred = a * t[idx] + b
    ss = 1 - np.sum((np.log(y[idx]) - pred) ** 2) / max(
        np.sum((np.log(y[idx]) - np.log(y[idx]).mean()) ** 2), 1e-30)
    return dict(v_dark=v_dark, step=step, tau=float(-1.0 / a), r2=float(ss))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Supplementary Figure 15: node capacitance versus discharge")
    ap.add_argument("--caps", nargs="*", default=["100nF", "10nF", "1nF"])
    ap.add_argument("--masks", nargs="*", default=["1111", "0000"])
    ap.add_argument("--last", type=int, default=0,
                    help="deprecated; kept so old command lines still run")
    ap.add_argument("--panel-spans", nargs=3,
                    default=["0,41.4", "10.6,24", "11.2,14"],
                    help="x-range of each of the three panels, e.g. 0,42")
    ap.add_argument("--no-tau", dest="with_tau", action="store_false", default=True,
                    help="omit tau from the legend")
    ap.add_argument("--figsize", nargs=2, type=float, default=None,
                    help="figure size in inches; default depends on the panel count (single panel 7.3x6.3)")
    ap.add_argument("--panels", default="abc",
                    help="which panels to draw: any combination of a / b / c (default abc)")
    ap.add_argument("--panel-letter", dest="panel_letter", default="",
                    help="force a panel letter even for a single panel, e.g. --panel-letter a")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args(argv)

    want = []
    for c in a.caps:
        m = CAP_RE.search(c)
        want.append(float(m.group(1)) * CAP_UNITS[m.group(2).lower()])
    want.sort(reverse=True)

    recs = [r for r in (load(row["run"]) for row in load_runs())
            if r and r["cap"] is not None and pF(r["cap"]) in CAP_COLOR
            and pF(r["cap"]) in [pF(c) for c in want]]

    picked = []
    for cap in want:
        for mk in a.masks:
            grp = [r for r in recs if pF(r["cap"]) == pF(cap) and r["mask"] == mk]
            good = [r for r in grp if r["ptp"] == r["ptp"] and abs(r["ptp"]) >= 0.2]
            if good:
                r = good[-1]
                r["an"] = tau_of(r)
                picked.append(r)
            elif grp:
                print(f"warning: no usable data for {cap_label(cap)} mask={mk} (skipped)")
    if not picked:
        print("no data found")
        return 1

    print("=" * 84)
    print("Figure data (tau: log-linear fit of |V - V_dark| after light-off; "
          "asymptote from the 1 s before light-on)")
    print("=" * 84)
    print(f"{'cap':>8} {'mask':>5} {'run':>18} {'R (MOhm)':>10} {'step mV':>9} "
          f"{'τ(s)':>9} {'R²':>7}")
    for r in picked:
        print(f"{cap_label(r['cap']):>8} {r['mask']:>5} {r['name']:>18} "
              f"{(r['res'] / 1e6 if r['res'] else float('nan')):>5.0f}M "
              f"{r['an']['step'] * 1e3:>+9.1f} {r['an']['tau']:>9.4f} "
              f"{r['an']['r2']:>7.4f}")

    spans = []
    for s in a.panel_spans:
        lo, hi = (float(x) for x in s.split(","))
        spans.append((lo, hi))
    sel = [c for c in "abc" if c in a.panels.lower()]
    if not sel:
        print("--panels must be a combination of a / b / c, e.g. a, ab, abc")
        return 1
    spans = [spans["abc".index(c)] for c in sel]
    n_panel = len(spans)
    single = (n_panel == 1)
    figsize = tuple(a.figsize) if a.figsize else ((7.3, 6.3) if single else
                                                 (11.5, 5.6) if n_panel == 2 else
                                                 (13.5, 5.5))

    fig, axes = plt.subplots(1, n_panel, figsize=figsize, squeeze=False)
    axes = list(axes[0])
    handles, labels = [], []
    # the legend fills column by column, so the order is derived from the
    # intended layout: multi-panel (ncol=4, 6+1 entries, two rows) puts all
    # 1111 in the first row and all 0000 in the second; single panel (ncol=2,
    # 6 entries, three rows) gives one capacitance per row, one mask per column
    mosort = {m: k for k, m in enumerate(a.masks)}
    if single:
        ordered = sorted(picked, key=lambda r: (mosort.get(r["mask"], 9), -pF(r["cap"])))
        ncol = 2
    else:
        ordered = sorted(picked, key=lambda r: (-pF(r["cap"]), mosort.get(r["mask"], 9)))
        ncol = 4
    for r in ordered:
        col = CAP_COLOR[pF(r["cap"])]
        ls = MASK_STYLE.get(r["mask"], "-")
        t = np.arange(r["v"].size) * r["dt"]
        lab = f"{cap_label(r['cap'])}, mask {r['mask']}"
        if a.with_tau:
            lab += f" ($\\tau$ = {r['an']['tau']:.3g} s)"
        for ax in axes:
            ax.plot(t, r["v"] * 1e3, color=col, ls=ls, lw=1.9,
                    solid_capstyle="round", label=lab, zorder=3)
        handles.append(Line2D([], [], color=col, ls=ls, lw=1.9))
        labels.append(lab)
    if not single:                     # for a single panel the grey band is labelled on the axes, not in the legend
        handles.append(Patch(facecolor="0.87", edgecolor="none"))
        labels.append("illumination on")

    w0 = picked[0]["wins"][0] if picked[0]["wins"] else None
    for k, (ax, (lo, hi)) in enumerate(zip(axes, spans)):
        if w0:
            ax.axvspan(w0[0], w0[1], color="0.87", lw=0, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(-600, 600)
        ax.set_box_aspect(1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Node voltage (mV)")
        ax.minorticks_on()
        for sp in ax.spines.values():
            sp.set_linewidth(2.0)
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.tick_params(which="major", length=4.0, width=0.9)
        ax.tick_params(which="minor", length=2.2, width=0.9)
        # panel letters carry no parentheses and sit at axes coordinates (-0.24, 1.02);
        # va=bottom keeps them fully above the axes box, clear of the top tick labels.
        # a single panel is unlabelled unless --panel-letter is given
        if n_panel > 1 or a.panel_letter:
            ax.text(-0.24, 1.02, a.panel_letter or "abc"[k], transform=ax.transAxes,
                    fontsize=30, fontweight="bold", va="bottom", ha="left")
        if single and w0:                # single panel: label the grey band on the axes
            ax.annotate("illumination on",
                        xy=((w0[0] + w0[1]) / 2.0, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", va="bottom", fontsize=12, color="0.35")

    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=ncol, handlelength=1.7, borderpad=.35, labelspacing=.25,
               columnspacing=1.4, handletextpad=.6)
    fig.tight_layout(h_pad=2.6, w_pad=3.6, rect=(0.0, 0.20, 1.0, 1.0))
    os.makedirs(OUTDIR, exist_ok=True)
    outs = []
    for ext in ("png", "svg", "pdf"):
        p = os.path.join(OUTDIR, f"{STEM}.{ext}")
        fig.savefig(p, dpi=a.dpi, facecolor="white")
        outs.append(p)
    plt.close(fig)

    # ---- caption and tau table (the caption describes the figure only; the
    # values belong in the main text or a Note) --------------------------------- #
    cap_txt = "/".join(cap_label(c) for c in want)
    cap_doc = {100e-9: "100 nF", 10e-9: "10 nF", 1e-9: "1 nF"}
    r_ohm = next((r["res"] for r in picked if r["res"]), None)
    panel_desc = {"a": "Full 41.4 s record. ",
                  "b": "Enlarged view around light-off. ",
                  "c": "Same interval on a shorter time scale. "}
    if single:
        first = ""
    else:
        first = " ".join(f"({c}) {panel_desc[c]}" for c in sel).strip() + " "
    lines = [
        "# Supplementary Figure 15 \u2014 caption",
        "",
        f"**Supplementary Figure 15 |** Time-domain photoresponse of the "
        f"floating-gate node with different external node capacitances. "
        f"{first}"
        f"The gate mask pattern is held at 1111 (solid lines) or 0000 "
        f"(dashed lines); the corresponding gate levels are "
        f"$V_\\mathrm{{GS}}$/$V_\\mathrm{{GD}}$ = −3 V/+3 V and +3 V/−3 V, "
        f"respectively. Colors denote the external node capacitance: "
        f"100 nF (blue), 10 nF (red) and 1 nF (orange). "
        f"The grey band marks the illumination window "
        f"({w0[0]:g}–{w0[1]:g} s). External discharge resistor "
        f"{r_ohm / 1e6:.0f} MΩ; laser drive current 80 mA; "
        f"sample interval 40 ms.",
        "",
        "## Fitted time constants",
        "",
        "| Node capacitance | Gate mask | τ (s) | R² |",
        "|---|---|---|---|",
    ]
    if not single:
        panel_lbl = {"a": "(a) Full 41.4 s record;",
                     "b": "(b) Enlarged view around light-off;",
                     "c": "(c) Same interval on a shorter time scale."}
        lines[4:4] = [
            "## Panel labels (in parentheses, matching the other captions)",
            " ".join(panel_lbl[c] for c in sel),
            "",
        ]
    for r in ordered:
        lines.append(f"| {cap_doc.get(r['cap'], cap_label(r['cap']))} | "
                     f"{r['mask']} | {r['an']['tau']:.3g} | {r['an']['r2']:.4f} |")
    lines += [
        "",
        "tau is obtained from a single-exponential fit of $|V - V_\\mathrm{dark}|$ after "
        "light-off, with the asymptote taken from the 1 s before light-on; "
        "R^2 is that of the log-linear fit.",
    ]
    cp = os.path.join(OUTDIR, f"{STEM}_caption.md")
    with open(cp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\noutput:")
    for p in outs:
        print("  " + p)
    print("  " + cp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
