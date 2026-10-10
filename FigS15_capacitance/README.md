# Supplementary Figure 15 — fading-memory timescale versus node capacitance

Light response of the reservoir node measured with three external node
capacitances. The figure is **Supplementary Figure 15** of the manuscript.

## Measured decay time constants

| Node capacitance | mask 1111 | mask 0000 |
|---|---|---|
| 100 nF | 6.977 s | 7.021 s |
| 10 nF | 0.738 s | 0.727 s |
| 1 nF | 0.155 s | 0.154 s |

These are the six values printed in the figure legend. τ is obtained from a
single-exponential fit of \|V − V_dark\| after light-off, with the asymptotic
level taken from the 1 s before light-on; the tabulated R² is that of the
log-linear fit and is ≥ 0.994 for all six traces.

The 100/10 ratio is 9.5 and the 10/1 ratio is 4.8: the 100 nF and 10 nF points
follow τ = R·C, while at 1 nF the parasitic node capacitance is no longer
negligible.

## Run

```bash
python plot_SI_capacitance_discharge.py --panels a   # -> figures/SI/ (png, svg, pdf, caption)
python check_tau_values.py                          # independent re-derivation of the six τ
```

`--panels a` is required: without it the script draws the three-panel variant,
which is a different figure. On a Windows console whose code page is GBK, set
`PYTHONIOENCODING=utf-8` first, otherwise the script aborts while printing "−"
(U+2212).

`check_tau_values.py` writes nothing and exits non-zero on any disagreement.

## Contents

```
.
├── README.md
├── requirements.txt
├── runs.csv                             the six runs: settings and fitted τ
├── run_manifest.py                      shared loader for runs.csv and the traces
├── plot_SI_capacitance_discharge.py     draws Supplementary Figure 15
├── check_tau_values.py                  independent verification of the six τ
├── mask_test_data/<run>/400/<mask>/1111/1111.csv    response traces
└── figures/SI/                          the delivered figure and its caption
```

Each response trace is a tab-separated, headerless four-column record; of its
four fields the second is the measured node voltage. The acquisition settings that define how each
trace is indexed — sampling interval, light window, and the derived settle and
active sample counts — are columns of `runs.csv`, so no per-run instrument log
is needed to read the data.

## Verification

| Check | Result |
|---|---|
| `check_tau_values.py` re-derives all six τ from the response traces | identical to `runs.csv` and to the figure legend to all printed digits |
| the rounded values quoted elsewhere (0.15 s, 0.73 s, ≈10×, not 10×) | all reproduced |
| delivered figure | `figures/SI/FigS_capacitance_discharge.png`, MD5 `92eba687a09a7730ff6b68a3ea251141` |

Re-running the plotting script regenerates the figure but not a byte-identical
PNG, because matplotlib records its version in the file metadata and renders
text slightly differently across versions. The content and the τ values are
unaffected, and the τ values are what `check_tau_values.py` verifies.

## Acquisition self-check flag

`runs.csv` carries the `valid` flag of each recording. The three `mask 0000`
runs are flagged `valid:0`: their `monitor_ptp_V` is only about 6 mV, so the
AO0 monitor channel recorded almost no swing and the coupling estimate derived
from it is not meaningful. The response channel itself is unaffected —
peak-to-peak ≈ 0.5 V, τ fitted with R² ≥ 0.994, and the sign is opposite to the
corresponding `mask 1111` run, as expected from the polarity reversal between
the two gate configurations.
