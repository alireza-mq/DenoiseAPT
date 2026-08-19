# External denoiser benchmark v1

## Purpose

This benchmark compares DenoiseAPT with independent time-series denoisers on
identical observations, clean references, windows, corruptions, evaluation
scaling, metrics, and scorer thresholds.  Method-specific preprocessing is
declared separately.  It defines an independent-denoiser benchmark rather than
an ablation and does not modify the completed Hybrid-v2 confirmation.

## Scientific status

The Sensor/Medical panel had already been opened when the external comparator
roster was chosen.  Therefore every new contrast in this benchmark is
**retrospective and descriptive**.  The output cannot be used as a new
confirmation result or as a universal state-of-the-art claim.

## Methods

- **Corrupted input** is a sanity reference, not a denoiser.
- **Median filter** uses a three-sample window and edge padding.
- **Wavelet thresholding** uses a development-selected wavelet, threshold rule,
  and multiplier through PyWavelets 1.8.0 (MIT).  Donoho's soft-thresholding
  paper is cited as a foundation, not as the exact selected adapter.
- **Noisereduce** uses the version 3.0.3 package (MIT).  Its stationary
  mode, reduction strength, and FFT size are selected on development data.
- **RINS-T adaptation** uses the official architecture at commit
  `95d1d9b44b44ba771b2400f5fb68fe42447c5fa2` and the demo-recipe
  hyperparameters, plus a declared observation-only min--max wrapper and a
  benchmark-owned optimization loop.  It is therefore an
  official-architecture/demo-recipe adaptation, not an unmodified official
  denoising program.  The
  repository contains no software license, so its checkout is excluded from
  public release artifacts unless permission or a license is obtained.
- **DenoiseAPT** is loaded byte-for-byte from `output__hybrid_v2` in the
  completed audit trace.  It is not retuned or rerun.

ECG-specific methods are deliberately excluded from the mixed-domain table.
Five windows labeled `Medical` come from SED engine-disk series, and the ECG
windows do not retain the sampling-rate and lead metadata needed for a faithful
published-checkpoint comparison.

## Development selection

One validation window per independent source group is chosen by a metadata-only
minimum-SHA-256 rule.  Each selected window receives Gaussian, impulse, drift,
and mixed corruption at severities 0.25, 0.50, and 0.75.  An equal-source-group
mean all-sample OS-NRMSE selects Wavelet and Noisereduce configurations.  Labels
and test outputs are not used.

## Evaluation

The retrospective panel contains 108 windows from 32 source groups, expanded
to 2,700 conditions including identity rows.  Reconstruction endpoints exclude
identity rows.  Conditions are averaged within source group, source groups are
averaged within Sensor or Medical, and the two domains receive equal weight.

The aggregate benchmark summary reports:

1. overall OS-NRMSE on corrupted conditions;
2. anomaly-region OS-NRMSE on labeled samples in corrupted event windows;
3. the percentage of observation-detectable labeled event-condition
   opportunities retained under configured Scorers A and B; and
4. output-only evidence intervals under those scorers.

The scorer outcomes are detector-specific checks, not physical anomaly truth.

## Lifecycle

Run the phases in order:

```text
python experiments/run_external_denoiser_benchmark_v1.py --tune
python experiments/run_external_denoiser_benchmark_v1.py --freeze
python experiments/run_external_denoiser_benchmark_v1.py --evaluate
python experiments/run_external_denoiser_benchmark_v1.py --audit
```

`--freeze` writes an exclusive receipt binding the development selection, code,
the benchmark dependency manifest and third-party notice, package versions,
model/runtime manifest, thresholds, RINS-T source files, and spent trace bytes
before external outputs are generated.  For Noisereduce, `sr=512` is only an
algorithmic index rate for a normalized 512-sample window; it is not a measured
physical sampling frequency.
