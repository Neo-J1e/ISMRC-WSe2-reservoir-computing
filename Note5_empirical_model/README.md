# Empirical SMRC model: reproducibility and validation package

This package reproduces the empirical-model parameters, fit diagnostics and continuous-sequence validation reported in Supplementary Tables 3–4 and Supplementary Figs. 8–9. The model logic follows the MATLAB Live Script archived in [source_reference/single_device_model_no_mask_diff_tao.mlx](source_reference/single_device_model_no_mask_diff_tao.mlx). Files under `data/` are byte-for-byte copies of the calibration records the script reads.

## One-command run

Open MATLAB R2023a (Update 5) with Curve Fitting Toolbox and run:

```matlab
run('run_reproducibility_audit.m')
```

The script fixes the audit seed to `rng(20260907,'twister')`. Continuous-sequence model validation is deterministic (`noise_sigma = 0`). All derived files are written to `outputs/`; no file under `data/` is modified.

The complete entry script was executed twice from the packaged data. The four key CSV outputs were byte-identical on both runs; their reference hashes are listed in [expected_output_hashes.csv](outputs/expected_output_hashes.csv).

The measured/model comparison for all 14 mask sequences is drawn from the
residuals the audit writes. The same data appears in two layouts, one for each
document that uses it:

```bash
python plot_14_mask_validation.py      # -> outputs/classification_continuous_validation_14masks.png
python plot_R7_14_mask_validation.py   # -> outputs/Figure_R7_14masks.png
```

The first is the square layout of Supplementary Fig. 9; the second is the
landscape layout of Figure R7 in the response to referees. Both need only numpy
and matplotlib.

## Exact empirical model reconstructed from the MLX

For mask polarity \(P_n\in\{-1,+1\}\), input drive \(I_n\), and \(\Delta t=0.2\,\mathrm{s}\), the code first selects the equilibrium target \(A_{P_n}(I_n)\) and time constant \(\tau_{P_n}(I_n)\):

\[
A_{+}(I)=a_{+}e^{b_{+}I}+c_{+}e^{d_{+}I},\qquad
A_{-}(I)=a_{-}e^{b_{-}I}+c_{-}e^{d_{-}I},
\]

\[
\tau_{+}(I)=u_{+}e^{v_{+}I},\qquad
\tau_{-}(I)=u_{-}e^{v_{-}I}.
\]

For \(I<40\,\mathrm{mA}\), the target is set to zero and the polarity-specific mean dark-decay time is used: \(\tau_{\mathrm{off},+}=6.79721597862\,\mathrm{s}\) and \(\tau_{\mathrm{off},-}=6.90082179229\,\mathrm{s}\). The state update is

\[
V_n=(A_n^{\ast}-V_{n-1})\left[1-e^{-\Delta t/\tau_n}\right]+V_{n-1}.
\]

The switching term is not added directly to \(V_n\). When the mask polarity reverses while the ordinary target is zero, the code replaces the target by

\[
A_n^{\ast}=\frac{P_n-P_{n-1}}{2\times0.3}.
\]

Thus the target-level switching magnitude is \(1/0.3=3.333333\ldots\,\mathrm{V}\), with sign set by the polarity change. After the discrete state update, the isolated switching contribution is approximately \(0.09665\,\mathrm{V}\) using \(\tau_{\mathrm{off},+}\) or \(0.09522\,\mathrm{V}\) using \(\tau_{\mathrm{off},-}\).

For the noise sweeps in the source MLX, independent Gaussian terms are added to both \(A\) and \(\tau\) as `sigma*randn`. The source variable is named `var`, but it functions as a standard deviation, not a variance.

## Model coefficients and diagnostics

- [Current-dependent coefficients and 95% CIs](outputs/functional_coefficients_and_ci.csv)
- [Goodness of fit for the four current-dependent functions](outputs/functional_fit_goodness.csv)
- [Residuals at all eight laser-current points](outputs/functional_fit_residuals.csv)
- [Per-current response/decay parameters](outputs/per_current_parameter_table.csv)
- [Single-pulse fit parameters, 95% CIs, R², adjusted R², RMSE, MAE, and bias](outputs/single_pulse_fit_parameters_and_diagnostics.csv)
- [Time-resolved single-pulse residuals](outputs/single_pulse_fit_residuals.csv)
- [Off-state parameters](outputs/off_state_parameters.csv)
- [Model constants](outputs/model_constants.csv)

The four current-dependent fits have R² values of 0.99978 (\(A_+\)), 0.96478 (\(\tau_+\)), 0.99971 (\(A_-\)), and 0.96479 (\(\tau_-\)). Across the 32 single-pulse response/decay fits, R² ranges from 0.97546 to 0.99988.

For the four current-dependent fits, the reported **fit RMSE** is `sqrt(SSE/(N-p))` over `N=8` current-wise points, with `p=4` for each amplitude function and `p=2` for each time-constant function. These values were calculated by the audit script; they are not erroneous but must not be confused with the continuous-sequence RMSE below, which uses the number of waveform samples as denominator. Neither scalar is a substitute for the corresponding pointwise residual file.

## Channel and mask definitions

- [Linear channel taps](outputs/channel_coefficients.csv): delays \(-7\) to \(+2\) with coefficients `[0.01, 0.03, 0.04, -0.05, 0.091, -0.10, 0.18, 1.00, -0.12, 0.08]`.
- [Channel nonlinearity](outputs/channel_nonlinearity.csv): \(I=q+0.03q^2-0.011q^3\).
- [All 14 four-bit masks](outputs/mask_definitions.csv): `0001` through `1110`, including one-, two-, and three-reversal words.

The channel taps and the memoryless nonlinearity used here are those implemented in the Figure 4 simulation: ten taps and \(I=q+0.03q^2-0.011q^3\), matching Supplementary Note 4.

## Continuous-sequence validation not used for the single-pulse fits

The parameter functions were obtained from `fig2_c.mat`. They were then frozen and evaluated on a measured continuous-input acquisition containing 14 masks and 8,000 state samples per mask (112,000 samples per dataset):

`fig2_c.mat` is the source **file name**, not a reference to panel 2c in the revised manuscript. The current-dependent single-pulse traces are shown in Supplementary Fig. 8.

| Dataset | Masks | Mean RMSE ± SD (V) | Mean range-normalized RMSE ± SD | Mean bias ± SD (V) | Mean Pearson r ± SD |
|---|---:|---:|---:|---:|---:|
| Classification, 2024-12-22 | 14 | 0.2130 ± 0.0595 | 0.2146 ± 0.0423 | −0.1457 ± 0.0920 | 0.7078 ± 0.1127 |

Detailed outputs:

- [Aggregate validation table](outputs/continuous_validation_aggregate.csv)
- [Classification per-mask metrics](outputs/classification_continuous_validation_metrics.csv)
- [Classification residuals](outputs/classification_continuous_validation_residuals.csv)
- [Classification overlays and residual distributions](outputs/classification_continuous_validation.png)
- [All 14 masks, measured versus model](outputs/classification_continuous_validation_14masks.png) — Supplementary Fig. 9
- [The same data in the landscape layout](outputs/Figure_R7_14masks.png) (file names keep the original Figure R7 label)

The model captures part of the temporal and polarity structure of the measured states but retains a systematic baseline and amplitude error; it is therefore used as a calibrated empirical trend model rather than as a pointwise predictor of the measured voltage.

## Measured gate-reversal transient used as a scale check

The byte-identical copy `data/Voc_all_400ms_no_wait.mat` and the read-only `verify_dark_switching_0000.m` extract gate-polarity reversals from all 16 mask traces recorded for zero optical code `0000`. At the 40 ms measurement interval, 128 reversals have median absolute adjacent-sample change 0.102087 V and interquartile range 0.006275 V; 127/128 voltage changes have the direction of the gate step. This is a measured total output transient, not a direct fit or pointwise validation of the model's fixed switching coefficient or its 0.2 s one-step contribution. Run `verify_dark_switching_0000` from MATLAB to reproduce the summary.

## Traceability and source-code audit

`source_reference/` contains a converted text copy of the new MLX and exact copies of the helper/fit functions found on the author workstation. They are included for inspection only and are not runnable here: they call helper folders that are not part of this package, and the wireless branch they still contain is not bundled. `run_reproducibility_audit.m` is the portable entry point used for all tables and validation results above.

Model scope and limitations:

- The original Live Script does not set a random seed; the packaged entry script fixes `rng(20260907,'twister')` so that its outputs are reproducible.
- The variant loop variable `tao_range` of the source script does not enter the state equation; the packaged entry script therefore uses a single fixed update step.
- The task readout weights are obtained by pseudoinverse on the same samples that are scored (in-sample), as also stated in the Supplementary Information; they are not held-out estimates.

## Data-copy manifest

| File | Bytes | SHA-256 |
|---|---:|---|
| `chip1-classification-20241222.mat` | 515,483 | `E8D58095D184577389ECA9BEA0B2FF914FDA1BD50A700C85E85ED5CE19505B54` |
| `fig2_c.mat` | 423,227 | `5B9017016A21362EE1C7728EF9B2552C788E2E7659236A516AABD2D9172F4720` |
| `single_device_model_data.mat` | 5,708 | `523ADF1CC2AF706EA2C959C273996038DB1782CB2D14E292CEE845B48D43DE22` |
| `Voc_all_400ms_no_wait.mat` | 2,374,254 | `118F52C699A620006A94EC2E04904660670A9E60709B233719ABBD8ED9800661` |
| `wave_classification_gt.txt` | 4,000 | `C7CCC9514DD44D4AF6241FB574F822132B7227A475E773B78D7660C4BE5F0C90` |
| `wave_classification_laser.txt` | 12,686 | `8C26D73F12AB7125D5218E05A060B2D6A3F2C1F7E8498650C164E0439C0B8427` |
