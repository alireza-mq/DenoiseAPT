# Public repository scope

This Git repository is a curated public software snapshot rather than a copy of
the full research workspace. Development history and sealed evaluation evidence
are retained separately by the authors.

## Included

- MIT-licensed DenoiseAPT application code and browser assets.
- Frozen live-runtime checkpoints and configuration.
- A deterministic synthetic fixture used only by API and installation tests.
- Two integrity-pinned demonstration replays: CATSv2 rich dynamics and MIT-BIH
  ECG anomaly preservation, under their separate upstream-data notices.
- Aggregate external-benchmark provenance and summary.
- Self-contained tests, setup scripts, and software documentation.

The revised browser hides the synthetic fixture from the benchmark selector. It
remains in the repository only so the public package can be checked without
redistributing an upstream dataset.

## Excluded from Git

- `.venv`, caches, logs, sessions, temporary files, and build products.
- Internal run directories and sealed condition-level evidence.
- Raw TSB-AD-U data, extracted upstream records, and every derived evaluation
  window except the two explicitly bundled demonstration replays.
- Per-window outputs, waveform rows, witness rows, and audit traces.
- The unlicensed RINS-T source checkout and unused third-party baselines.
- Literature libraries, internal workflow state, superseded figures, and old
  release archives.

The browser exposes only the two bundled benchmark presets and compatible CSV
inspection in review-only mode; it does not substitute the hidden synthetic
test fixture. These exclusions prevent accidental
redistribution of upstream data, private research traces, unlicensed
third-party source, and obsolete material. They do not imply that the retained
research archive was deleted.

The MIT License applies to the included DenoiseAPT code. The CATSv2 and MIT-BIH
replays remain under the separate terms recorded in `LICENSES/`.
