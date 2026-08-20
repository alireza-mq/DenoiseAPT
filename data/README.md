# Demo data

The revised application separates benchmark replays from local CSV inspection.
A deterministic synthetic fixture is retained only for automated
tests and is not a user-facing data source.

## Demonstration replays

The main workflow uses a 512-time-point CATSv2 simulated telemetry window. A
local MSL Sensor replay provides the sparse anomaly-preservation spotlight used
in the recording. The bundled MIT-BIH Arrhythmia Database ECG window is the
optional third case. Each replay contains a derived source window, its
evaluation-only reference and labels, and frozen matched denoiser outputs for a
controlled corruption condition.

The MSL manifest is published for integrity and provenance, but the exact MSL
data artifact is excluded until its upstream redistribution terms are explicit.

These windows were selected after panel inspection for visual explanation and
must not be treated as additional aggregate benchmark evidence. Each artifact
is accepted only with its integrity manifest and pinned provenance hashes; a
missing or mismatched manifest fails closed.

## CSV review-only mode

The browser accepts a headered, comma-delimited file with one finite numeric
signal column. Conventional time columns and numeric `label`, `anomaly`, or
`target` columns are optional. The service supports a univariate series of
64--250,000 points and analyzes one contiguous interval of 64--2,048 points.

CSV values are treated as the observation. The upload path performs no
resampling, missing-value imputation, multivariate modeling, or cross-domain
calibration. Without a known pre-corruption reference, reconstruction measures
are unavailable. Labels, when present, are used only for display. An upload is
always review-only and never receives held-out replay or benchmark-certificate
status.

## Hidden test fixture

`data/prepared/synthetic_guided_case.npz` is deterministic data created by the
DenoiseAPT authors for API, installation, and UI regression tests. The revised
browser filters it out of the benchmark selector. It is not a TSB-AD record,
does not support benchmark claims, and must not be presented as the guided
demonstration.

The fixture can be regenerated without downloading TSB-AD:

```bash
python scripts/download_data.py --fixture-only --destination data/fixture
```

## Provenance and licensing

- Official archive: <https://www.thedatum.org/datasets/TSB-AD-U.zip>
- TSB-AD project: <https://github.com/TheDatumOrg/TSB-AD>
- Dataset-specific sources and licenses:
  <https://thedatumorg.github.io/TSB-AD/#summary-of-datasets>

The TSB-AD project states that its preprocessing and curation code is released
under Apache-2.0, while constituent datasets retain their original terms. The
CATSv2 replay is governed by CC BY 4.0, and the MIT-BIH replay is governed by
ODC Attribution 1.0. Exact attribution and license links are in
`LICENSES/CATS-DATA-NOTICE.md` and `LICENSES/MITDB-DATA-NOTICE.md`; neither
artifact is covered by the project MIT license.
