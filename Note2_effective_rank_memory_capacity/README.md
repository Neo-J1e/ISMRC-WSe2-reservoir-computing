# Supplementary Note 2 — Effective dimensionality and memory capacity

Regenerates every number quoted in **Supplementary Note 2** and the panels of
**Supplementary Figure 5**.

## Run

```bash
python analyze_effective_rank_mc.py            # analysis + self-check
python analyze_effective_rank_mc.py --no-check # analysis only
python analyze_effective_rank_mc.py --data-root /path/to/data --output-dir /tmp/out
```

No absolute path is hard-coded. Everything is resolved relative to this file,
so the folder can be copied anywhere.

## Inputs (`./data`)

| Path | Content |
|---|---|
| `multi_states/2025-01-02-23-08/400/` | the state-space measurement behind Supplementary Fig. 5: 16 mask words × 16 optical inputs, mask-interval duration 400 ms |
| `matlab/chip1-classification-20241129.mat` | reservoir state matrix, 14 time-varying masks, `data_all` (14, 8000) |
| `matlab/chip1-classification-20250101.mat` | fixed-gate state matrix, `data_all` (11, 8000); rows 5 and 6 are 0000 and 1111 |
| `matlab/20240818/wave_classification_laser.txt` | 2000-step multi-level input waveform |
| `matlab/20240818/wave_classification_gt.txt` | its labels |

Each recording under `multi_states/` is a tab-separated file with no header; of
its four fields the second is `V_OC(t)` in volts and the fourth is `V_g(t)` in
volts.

> **Note on the `.mat` files.** Only the files that contain a `data_all`
> state matrix are bundled.

## Outputs (`./outputs`)

| File | Used for |
|---|---|
| `note2_effrank_by_optical_input.csv` | Supplementary Fig. 5a |
| `note2_effrank_by_mask_word.csv` | Supplementary Fig. 5c |
| `note2_memory_capacity.csv` | Supplementary Fig. 5d |
| `note2_dimension_table.csv` | the per-session table printed by the script |
| `note2_results.json` | all scalars, including the singular-value spectrum for Fig. 5b |

## Verified results

Ran with Python 3.11.7 / numpy 2.4.6 / scipy 1.17.1. The self-check compares
every quantity below with the value printed in Supplementary Note 2:

| Quantity | Regenerated | In SI |
|---|---|---|
| optical-input-indexed effective rank, min / max / mean | 3.80 / 7.43 / 6.06 | 3.80 / 7.43 / 6.06 |
| mask-indexed effective rank over the 14 masks, min / max / mean / SD | 5.01 / 8.36 / 6.37 / 1.00 | 5.01 / 8.36 / 6.37 / 1.00 |
| fixed-gate effective rank, 0000 / 1111 | 3.25 / 3.37 | 3.25 / 3.37 |
| numerical rank, fixed gate / time-varying masks | 5–6 / 11–14 | 5–6 / 11–14 |
| memory capacity per delay, masked | +0.45 ± 0.08 | +0.45 ± 0.08 |
| memory capacity summed over one period, masked / 0000 / 1111 | +3.59 / −2.80 / −3.14 | +3.59 / −2.80 / −3.14 |
| nonlinear capacity, masked / fixed gate | +0.01 / −0.03 | +0.01 / −0.03 |

All 18 checks pass.

## Method

1. **Virtual-node extraction.** The gate trace is rounded, intermediate levels
   are repaired, transitions are located, and every transition is refined to the
   local `V_OC` extremum imposed by the mask bit. Fixed mask words (0000, 1111)
   contain no transition and use a uniform grid over the same window, so masked
   and fixed-gate conditions always share the same sampling instants and the same
   number of readout coordinates.
2. **Effective rank.** The state matrix is centred column-wise and the
   participation ratio `(Σσ)² / Σσ²` is evaluated on its singular values.
3. **Memory capacity.** Ridge readout (`λ = 1e-3`), chronological 70/30
   train/test split, `MC_k = R²` for reconstructing `u(t−k)` from the state at
   step `t`. Because the input is periodic with period 8, the values are averaged
   per input cycle. The values quoted in the SI are those of the **first** input
   period; the script reports the later periods as well.
