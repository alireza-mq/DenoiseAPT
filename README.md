# DenoiseAPT

DenoiseAPT is a local research demonstration for generative time-series
denoising with configured anomaly-evidence checks. It combines a fixed signal
filter, a learned repair generator, two frozen scorers, evidence-guided routing,
and reversible interval editing.

This public repository contains the MIT-licensed application, frozen runtime
artifacts, tests, and aggregate provenance for the retrospective denoiser
benchmark. Repository: <https://github.com/alireza-mq/DenoiseAPT>.

## Demo video

**[▶ Watch the DenoiseAPT ICDM 2026 demonstration online](https://alireza-mq.github.io/DenoiseAPT/)**

The revised video presents matched denoiser comparison, configured anomaly
preservation evidence, expert guided interval adaptation, and the sparse
anomaly preservation spotlight with synchronized, timed on-screen guidance.
English captions are included with the release.

[MP4 video](https://github.com/alireza-mq/DenoiseAPT/releases/download/icdm-2026-demo-video-v1/DenoiseAPT_ICDM_2026_Demo_Video.mp4) ·
[Captions (SRT)](https://github.com/alireza-mq/DenoiseAPT/releases/download/icdm-2026-demo-video-v1/DenoiseAPT_ICDM_2026_Demo_Captions.srt)

## Public repository contents

- `denoiseapt/`, `server.py`, and `web/`: local application and browser UI.
- `checkpoints/automatic_preservation/`: frozen seed-17 runtime artifacts.
- `data/prepared/synthetic_guided_case.npz`: deterministic fixture retained
  only for API and installation tests; the revised browser hides it from the
  benchmark selector.
- `data/prepared/*_replay.npz`: two small, integrity-pinned bundled replays
  governed by the separate notices in `LICENSES/`.
- `data/prepared/tsb_ad_msl_heldout_replay_manifest.json`: integrity metadata
  for the optional local MSL spotlight; its data artifact is not redistributed.
- `reproducibility/`: frozen configuration, receipts, code, and aggregate
  summary for the external-denoiser benchmark.
- `tests/`: self-contained application and metric tests.

The repository deliberately excludes raw benchmark records, all other derived
evaluation windows, condition-level traces, internal run directories,
environments, third-party source trees, and historical material. See
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

The browser opens with `CATS rich dynamics`. When the integrity-matched local
artifact is installed, `MSL anomaly preservation` appears second as the sparse
spotlight used in the five-minute recording. `MITDB ECG` remains available as
the bundled optional case. The packaged synthetic fixture remains available to
automated tests but is not shown as a demonstration case.

## Bundled held-out benchmark replays

The main workflow uses an integrity-pinned, 512-time-point CATSv2 simulated
telemetry window. The local MSL spotlight is almost flat except for one narrow
event: Median Filter and RINS-T remove it, NoiseReduce attenuates it, Wavelet
retains it with the injected impulses, and Our Model retains it while achieving
the lowest displayed error. The bundled MIT-BIH ECG replay is the optional
third case. Each stores one derived evaluation window and frozen, matched
outputs for the same controlled corruption condition. The examples were
selected after the benchmark panel was examined; they are illustrative and do
not replace the aggregate benchmark.

The bundled replays have integrity manifests and separate upstream-data
notices; the service fails closed if an artifact or provenance hash does not
match. The MSL artifact stays local until its upstream data redistribution
terms are explicit. Replay data are not covered by the project MIT license. See
[`data/README.md`](data/README.md) and
[`docs/DATA_CARD.md`](docs/DATA_CARD.md).

For either replay, the comparison view contains only:

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

These self-contained checks cover the hidden deterministic fixture, the two
bundled replays, and the optional MSL manifest boundary. They do not turn any
selected window into aggregate benchmark evidence.

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

The DenoiseAPT code is released under the MIT License. See [`LICENSE`](LICENSE)
and [`LICENSES/THIRD_PARTY_NOTICES.md`](LICENSES/THIRD_PARTY_NOTICES.md).
The two replay artifacts are governed by
[`LICENSES/CATS-DATA-NOTICE.md`](LICENSES/CATS-DATA-NOTICE.md) and
[`LICENSES/MITDB-DATA-NOTICE.md`](LICENSES/MITDB-DATA-NOTICE.md), not the
project MIT license.
