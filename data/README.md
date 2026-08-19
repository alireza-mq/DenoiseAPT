# Demo data

The revised application separates an optional benchmark replay from local CSV
inspection. A deterministic synthetic fixture is retained only for automated
tests and is not a user-facing data source.

## Optional TSB-AD-U CATSv2 replay

The local benchmark demonstration uses a 512-point held-out window from CATSv2,
a simulated complex dynamical-system telemetry dataset included in TSB-AD-U. The
replay contains a derived source window, its evaluation-only reference and
labels, and frozen matched denoiser outputs for one controlled corruption
condition. It was selected after panel inspection for visual explanation and
must not be treated as additional aggregate benchmark evidence.

The replay and its source records are intentionally absent from this public Git
repository pending review of upstream redistribution terms. An authorized local
copy is accepted only with its integrity manifest and pinned provenance hashes;
a missing or mismatched manifest must fail closed. The public checkout therefore
shows no benchmark preset and directs the user to CSV review-only mode.

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
under Apache-2.0, while constituent datasets retain their original terms. Its
dataset summary identifies CATSv2 as simulated telemetry and lists CC BY 4.0.
The same summary lists no license for UCR; absence of a stated license is not
permission to redistribute it. Consult and cite the original source before
publishing or redistributing any dataset-derived artifact.
