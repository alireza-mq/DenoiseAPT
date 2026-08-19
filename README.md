# DenoiseAPT

DenoiseAPT is a local research demonstration for generative time-series
denoising with configured anomaly-evidence checks. It combines a fixed signal
filter, a learned repair generator, two frozen scorers, evidence-guided routing,
and reversible interval editing.

This public repository contains the MIT-licensed application, frozen runtime
artifacts, tests, and aggregate provenance for the retrospective denoiser
benchmark. Repository: <https://github.com/alireza-mq/DenoiseAPT>.

## Public repository contents

- `denoiseapt/`, `server.py`, and `web/`: local application and browser UI.
- `checkpoints/automatic_preservation/`: frozen seed-17 runtime artifacts.
- `data/prepared/synthetic_guided_case.npz`: deterministic fixture retained
  only for API and installation tests; the revised browser hides it from the
  benchmark selector.
- `reproducibility/`: frozen configuration, receipts, code, and aggregate
  summary for the external-denoiser benchmark.
- `tests/`: self-contained application and metric tests.

The repository deliberately excludes raw benchmark records, derived evaluation
windows, condition-level traces, internal run directories, environments,
third-party source trees, and historical material. See
[`docs/ARCHIVE_SCOPE.md`](docs/ARCHIVE_SCOPE.md).

## Quick start

Python 3.10, 3.11, or 3.12 is required.

### Windows

```powershell
.\setup.ps1
.\run_demo.ps1
```

### Linux or macOS

```bash
chmod +x setup.sh run_demo.sh
./setup.sh
./run_demo.sh
```

Open <http://127.0.0.1:8765>. The service is intended for local use and has no
authentication layer, so do not bind it to a public network.

The public checkout does not include a benchmark replay. This is expected: the
benchmark selector remains empty and the interface directs the user to
**Upload CSV · review only**. The packaged synthetic fixture is still available
to automated tests but is not shown as a demonstration case.

## Optional held-out benchmark replay

The local benchmark demonstration uses an integrity-pinned, 512-point replay
drawn from a held-out TSB-AD-U CATSv2 simulated-telemetry window. It stores one derived
evaluation window and frozen, matched outputs for the same controlled
corruption condition. The example was selected after the benchmark panel was
examined to make the interaction legible; it is illustrative and does not
replace the aggregate benchmark.

That derived replay is intentionally omitted from public Git while upstream
redistribution terms are reviewed. An authorized local copy must include its
integrity manifest; the service fails closed if the artifact or provenance
hashes do not match. The omission is not an invitation to reconstruct or
redistribute the window without checking the constituent dataset terms. See
[`data/README.md`](data/README.md) and
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

When the replay is installed locally, the revised comparison view contains
only:

- Reference before corruption (evaluation only)
- Corrupted Observation
- Median Filter (w=3)
- Wavelet Thresholding
- Noisereduce
- RINS-T Adaptation
- Our Model

The evidence view compares only Corrupted Observation and Our Model. External
comparator traces are frozen matched outputs rather than models executed during
the browser request. Controlled corruption, the pre-corruption reference, and
event labels exist only for demonstration and evaluation; they are not normal
inference inputs.

The editable session begins from Our Model. An observation weight changes only
the selected interval, after which the service recomputes evidence and records
a fresh full-window A/B check. A passed check means only that the two configured
frozen scorers satisfied their recorded observation-component conditions. It
does not establish physical anomaly truth, detector-independent preservation,
clinical validity, or production readiness.

## CSV review-only mode

Upload CSV is a compatibility and inspection path, not a claim that the frozen
model generalizes to arbitrary unseen domains. The browser accepts a headered,
comma-delimited, univariate numeric series with an optional time column and an
optional numeric label column. It processes one contiguous time window locally.

Uploaded values are treated as the observation. Unless a separate controlled
workflow provides a pre-corruption reference, reference-based reconstruction
measures are unavailable. Uploads receive no held-out status, cross-domain
calibration, automatic benchmark certificate, resampling, missing-value
imputation, or multivariate modeling. Details are in
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Tests

After setup:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

For the end-to-end smoke test, keep `run_demo.ps1` running in one terminal and
use a second terminal:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

These self-contained checks use the hidden deterministic fixture and do not
turn it into benchmark evidence. Tests that require non-public evaluation data
remain unavailable unless the corresponding artifacts are installed locally
under their upstream terms.

## Reproducibility boundary

`reproducibility/external_denoiser_benchmark_v1/` preserves the benchmark
configuration, runner, dependency manifest, development selection, pre-freeze
and recovery receipts, and aggregate result summary. Re-execution requires
locally prepared benchmark data and dependencies under their upstream terms.

The RINS-T row is an official-architecture/repository-recipe adaptation. The
pinned upstream checkout did not contain a software license, so its source is
not redistributed in this repository.

## Citation

Citation metadata for this software release is provided in `CITATION.cff`.
Please also cite the datasets used in any downstream evaluation.

## License and third-party material

The DenoiseAPT code is released under the MIT License. See [`LICENSE`](LICENSE),
[`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md), and
[`LICENSES/APACHE-2.0-THIRD-PARTY.txt`](LICENSES/APACHE-2.0-THIRD-PARTY.txt).
Dataset terms remain with their upstream sources.
