# Supplementary Code and Data

Code and data accompanying

> **Electrically Programmed In-Sensor Masking Reservoir Computing Based on
> Reconfigurable WSe₂ Photodiodes**

Everything below regenerates the values and figures printed in the
Supplementary Information from the bundled measurements. No absolute path is
required: every script resolves its inputs relative to its own location, and
every entry point can be pointed elsewhere with `--data-root` / `--output-dir`.

## Contents

| Directory | Covers | Language | Entry point |
|---|---|---|---|
| [`Note2_effective_rank_memory_capacity/`](Note2_effective_rank_memory_capacity) | Supplementary Note 2, Supplementary Fig. 5 | Python | `analyze_effective_rank_mc.py` |
| [`Note5_empirical_model/`](Note5_empirical_model) | Supplementary Notes 3–5, Supplementary Figs. 8–9 | MATLAB, Python | `run_reproducibility_audit.m`, `plot_14_mask_validation.py`, `plot_R7_14_mask_validation.py` |
| [`Note6_repeatability_stability_noise/`](Note6_repeatability_stability_noise) | Supplementary Note 6, Supplementary Figs. 10–14, plus the raw acquisition data | Python | `analyze_note6.py`, `check_raw_provenance.py` |
| [`FigS15_capacitance/`](FigS15_capacitance) | Supplementary Fig. 15 | Python | `plot_SI_capacitance_discharge.py`, `check_tau_values.py` |

Each directory has its own README describing its inputs, outputs, method and
verified results.

## Reproducing the analysis

### Python packages

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

python run_all.py               # runs every Python package and self-checks it
```

`run_all.py` exits non-zero if any package fails to run or if any regenerated
value disagrees with the number printed in the Supplementary Information. The
packages can equally be run on their own; see their READMEs.

It runs the *verification* entry points only. The figure-drawing scripts are
not invoked, because they write into tracked directories; run them by hand as
described in the relevant README.

### MATLAB package

MATLAB R2023a (Update 5) or newer with the **Curve Fitting Toolbox** and the
**Statistics and Machine Learning Toolbox**:

```matlab
cd Note5_empirical_model
run('run_reproducibility_audit.m')
verify_dark_switching_0000
```

The audit script writes its derived tables and figures to
`Note5_empirical_model/outputs/`. Four of those outputs are checked against the
reference SHA-256 digests in `outputs/expected_output_hashes.csv`; the shipped
copies were confirmed byte-identical on a re-run from a different directory.

## Verification summary

| Package | Check | Result |
|---|---|---|
| Note 2 | 18 quantities compared with Supplementary Note 2 | all pass |
| Note 5 | 4 key CSVs compared with reference SHA-256 digests | all byte-identical |
| Note 5 | `verify_dark_switching_0000` summary | 16 traces, 128 reversals, 0.102087 V median, 0.006275 V IQR, 127/128 sign agreement — as printed in the SI |
| Note 6 | 46 quantities compared with Supplementary Note 6 | all pass |
| Note 6 | raw acquisition files → derived data (`check_raw_provenance.py`) | 451 files SHA-256 verified; 450 recordings identical to the raw CSV column 2; 10 Device 4 calibration traces identical to the raw light-on window |
| Fig. S15 | all 6 decay time constants re-derived from the raw traces and compared with the values printed in the figure legend | identical; all pass |

## What the data is

### Note 2 — effective rank and memory capacity (Supplementary Fig. 5)

The measurement applies a 4-bit mask word to the two gates of a split-gate WSe₂
node while a 4-bit optical pattern illuminates the device, and records the
resulting node voltage.

| Data | Content |
|---|---|
| `data/multi_states/2025-01-02-23-08/400/` | 256 recordings (16 mask words × 16 optical patterns), laid out as `<mask>/<optical pattern>/<pattern>.csv`. Each is 1160 rows × 4 tab-separated fields: field 2 is `V_OC(t)` in volts, field 4 is `V_g(t)`. Mask-interval duration 400 ms, gate amplitude ±2 V |
| `data/matlab/chip1-classification-20241129.mat` | variable `data_all`, (14, 8000): 14 time-varying-mask recordings, the state matrix of the memory-capacity analysis |
| `data/matlab/chip1-classification-20250101.mat` | variable `data_all`, (11, 8000); rows 5 and 6 are the fixed-gate 0000 and 1111 baselines |
| `data/matlab/20240818/wave_classification_laser.txt`, `wave_classification_gt.txt` | the 2000-step input waveform and its labels |

### Note 5 — empirical device model (Supplementary Notes 3–5, Figs. 8–9)

| Data | Content |
|---|---|
| `data/fig2_c.mat` | `all_data.p_voc` / `n_voc`: single-pulse response traces at eight laser currents (40–110 mA) in both polarities — the input to every coefficient of the empirical model |
| `data/single_device_model_data.mat` | the fitted per-current amplitudes and time constants with their 95 % confidence intervals (Supplementary Table 3) |
| `data/chip1-classification-20241222.mat` | variable `data_all`, (14, 8000): an independent continuous classification record, not used for the single-pulse fits, against which the frozen model is validated (Supplementary Fig. 9) |
| `data/Voc_all_400ms_no_wait.mat` | the zero-optical-code (0000) record covering all 16 mask words, from which the gate-reversal transient is measured |
| `data/wave_classification_laser.txt`, `wave_classification_gt.txt` | input waveform and labels of the classification task |

### Note 6 — repeatability, stability, parameter variation (Supplementary Figs. 10–14)

Raw acquisition, under `data/raw_acquisition_D2DC2C/`:

| Data | Content |
|---|---|
| `02_五器件_全部16种mask_400条/` | 400 recordings: five devices × 16 mask words × 5 cycles, laid out as `<run>/400/<mask>/1111/1111.csv` (`1111` is the fixed optical input code). Each is 1210 rows × 4 tab-separated fields; field 2 is the measured response voltage in volts |
| `01_恒定mask_50条_Note6标定用/` | the 50 constant-mask recordings (5 devices × {0000, 1111} × 5 cycles) behind the parameter-variation calibration |
| `清单_SHA256.csv` | relative path, byte size and SHA-256 of all 451 bundled files, plus the device, cycle and mask label of each |
| `00_数据说明.md` | packaging note, including a ready-to-use English Data Availability paragraph |

Derived data:

| Data | Content |
|---|---|
| `five_device/traces.npz`, `five_device/c4d4/C4D4_原始响应.npz` | the reviewed response traces of the five devices, as arrays |
| `five_device/alignment/` | the visually reviewed response-knee index of every recording and the candidate edges it was chosen from |
| `parameter_variation/Device4_10条原始与拟合数据.csv`, `Device4_逐cycle拟合参数.csv` | the ten Device 4 calibration traces and their per-cycle fits (Supplementary Figs. 12–13) |
| `parameter_variation/parameter_repeat_summary.csv`, `all_fits_and_window_checks.csv`, `protocol_trace_qc.csv` | the 50 constant-mask records and their fits |
| `stability/stability_8000steps.json` | the block statistics of the 8000-step record (Supplementary Fig. 11) |
| `noise_sweep/fig_noise_sweep_CI_*_raw.csv`, `fig_noise_sweep_CI_*.csv` | the 20 realisations per perturbation level and their means with 95 % confidence intervals (Supplementary Fig. 14) |

### Fig. S15 — fading-memory timescale versus node capacitance (Supplementary Fig. 15)

The same node measured with three external capacitances, each under the two
gate-mask configurations.

| Data | Content |
|---|---|
| `mask_test_data/<run>/400/<mask>/1111/1111.csv` | six response traces (1, 10 and 100 nF × mask 1111 and 0000), each 1035 rows × 4 tab-separated fields; field 2 is the node voltage |
| `runs.csv` | per run: capacitance, series resistance, mask, illumination window, sampling interval, the derived settle and active sample counts, step amplitude, and the `valid` self-check flag — together with the fitted τ and R² |

Acquisition parameters for every dataset are stated in the Methods, the
Supporting Information and the response to referees. The per-package READMEs
describe the format, method and verification of each package in full.

## Environment

Python packages were verified end to end on Python 3.11.7 (Windows) with

```
numpy 2.4.6   scipy 1.17.1   pandas 3.0.6   matplotlib 3.11.2
```

as pinned in `requirements.txt`. Python ≥ 3.10 is required by pandas 3.x.
The MATLAB scripts were verified on MATLAB R2023a (Update 5).

The Supplementary Fig. 15 package needs only numpy and matplotlib.

`.gitattributes` pins LF line endings across the repository, so a fresh
checkout reproduces the files exactly and the SHA-256 manifest of the raw
acquisition data stays valid on every platform; without it a Windows checkout
would rewrite line endings and invalidate every digest.

## Licence and contact

Please cite the article. Correspondence should be addressed to the
corresponding authors listed in the manuscript.