# Release scope and local archive

This Git repository is a curated public software snapshot, not a copy of the full
research workspace. The original workspace is retained locally as the archive
of development history, sealed evidence, literature, screenshots, and earlier
research drafts.

## Included

- Runnable DenoiseAPT application code and browser assets.
- Frozen live-runtime checkpoints and configuration.
- Synthetic demonstration data.
- Aggregate external-benchmark provenance and summary.
- Self-contained tests and setup scripts.

## Excluded from Git

- `.venv`, caches, logs, sessions, temporary files, and build products.
- `runs/` and other sealed or internal evidence directories.
- Raw TSB-AD-U data, extracted upstream records, and derived experiment NPZs.
- Per-window benchmark outputs, waveform rows, witness rows, and audit traces.
- The unlicensed RINS-T source checkout and unused third-party baselines.
- Historical drafts, reference examples, literature libraries, writing guides,
  prompt/workflow state, and superseded figures.
- Old release archives and typesetting auxiliary files.

These exclusions prevent accidental redistribution of upstream data, private
research traces, unlicensed third-party source, and obsolete material. They do
not imply that the excluded history was deleted from the author's local archive.
