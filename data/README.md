# Demo data

The package supports two deliberately separate data modes.

## Guided TSB-AD-U case

The default downloader selects:

`TSB-AD-U/442_UCR_id_140_Medical_tr_1875_1st_4187.csv`

This official TSB-AD-U series has 7,501 samples and a labelled interval at
source indices `[4187, 4199)`. The preparation script extracts a 512-sample
window centred on that event and records the exact source indices in the
output metadata.

Neither the archive nor this prepared guided NPZ is bundled in the release ZIP.
The commands below create it locally. Until then, the packaged synthetic case
remains runnable in explicit review-only mode.

Run from the package root:

```bash
python scripts/download_data.py
python scripts/prepare_demo_case.py --ensure-demo-case
```

The archive is cached in `data/raw/TSB-AD-U.zip` and validated against the
pinned SHA-256 in `dataset_metadata.json`. The extracted dataset and derived
guided NPZ are not committed or redistributed by this package.

The release validation completed with normal TLS verification and the pinned
checksum. The downloader fails closed if either check fails. An explicitly
insecure TLS mode exists only for controlled recovery and requires a non-empty
trusted SHA-256; it emits a warning and records the acquisition mode. Do not use
it simply to bypass a download error.

## Offline fixture

For fixture generation without downloading TSB-AD:

```bash
python scripts/download_data.py --fixture-only --destination data/fixture
```

The built-in fixture is deterministic synthetic data created by DenoiseAPT.
It is suitable for tests and UI development, but it is **not** a TSB-AD case
and must not be reported as benchmark evidence.

## Provenance and licensing

- Official archive: <https://www.thedatum.org/datasets/TSB-AD-U.zip>
- TSB-AD project: <https://github.com/TheDatumOrg/TSB-AD>
- Dataset-specific sources/licenses: <https://thedatumorg.github.io/TSB-AD/#summary-of-datasets>

TSB-AD states that its preprocessing and curation are released under
Apache-2.0. The datasets inside the collection retain their upstream terms.
The official page lists no license for UCR; absence of a stated license is not
permission to redistribute it. Users must consult and cite the original source
before publication or redistribution.
