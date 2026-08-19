# DenoiseAPT

DenoiseAPT is a local research demonstration for generative time-series
denoising with configured anomaly-evidence checks. It combines a fixed signal
filter, a learned repair generator, two frozen scorers, evidence-guided routing,
and reversible interval editing.

This repository contains the runnable demonstration, frozen runtime artifacts,
and aggregate provenance for the retrospective denoiser benchmark.

Repository: <https://github.com/alireza-mq/DenoiseAPT>

## Repository contents

- `denoiseapt/`, `server.py`, and `web/`: local application and browser UI.
- `checkpoints/automatic_preservation/`: frozen seed-17 runtime artifacts.
- `data/prepared/synthetic_guided_case.npz`: redistributable synthetic fixture.
- `reproducibility/`: frozen configuration, receipts, code, and aggregate summary
  for the external-denoiser benchmark.
- `tests/`: self-contained application and metric tests.

The repository deliberately excludes raw benchmark records, derived evaluation
windows, per-condition traces, internal run directories, Python environments,
historical drafts, and third-party source trees. See
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

Open <http://127.0.0.1:8765>. The default setup uses the packaged synthetic
fixture and does not download research data. The service is intended for local
use and exposes no authentication layer; do not bind it to a public network.

## Guided benchmark case

The optional guided case is prepared locally from the upstream TSB-AD-U archive:

```powershell
.\setup.ps1 -DownloadDataset
```

The downloader verifies TLS and a pinned SHA-256 value. The source archive and
prepared UCR-Medical window are not redistributed here. Without that exact
allowlisted window, the application remains usable in review-only mode.

## What the interface shows

For one observation, the interface compares a width-9 moving-average filter, a
matched conditional GAN, a preservation-trained generative repair candidate, a
dashboard routing comparison, and the established live controller output. The
editable session starts from the live controller output. The offline finite-grid
benchmark configuration is separate and is not the live automatic default.

The displayed concern cue is an inspection aid, not an anomaly probability. A
passed two-scorer record means only that the configured Scorers A and B satisfied
their frozen observation-component checks. It does not establish physical
anomaly truth, detector-independent preservation, clinical validity, or
production readiness.

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

The synthetic workflow runs without external datasets. Tests that require the
optional guided case remain unavailable until it is prepared locally.

## Reproducibility boundary

`reproducibility/external_denoiser_benchmark_v1/` preserves the exact benchmark
configuration, runner, dependency manifest, development selection, pre-freeze
and recovery receipts, and aggregate result summary. Re-executing it requires
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
