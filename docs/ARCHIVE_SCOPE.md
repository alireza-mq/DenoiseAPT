# Public repository scope

This Git repository is a curated public software snapshot rather than a copy of
the full research workspace. Development history and sealed evaluation evidence
are retained separately by the authors.

## Included

- MIT-licensed DenoiseAPT application code and browser assets.
- Frozen live-runtime checkpoints and configuration.
- A deterministic synthetic fixture used only by API and installation tests.
- Aggregate external-benchmark provenance and summary.
- Self-contained tests, setup scripts, and software documentation.

The revised browser hides the synthetic fixture from the benchmark selector. It
remains in the repository only so the public package can be checked without
redistributing an upstream dataset.

## Excluded from Git

- `.venv`, caches, logs, sessions, temporary files, and build products.
- Internal run directories and sealed condition-level evidence.
- Raw TSB-AD-U data, extracted upstream records, and derived evaluation windows.
- The integrity-pinned CATSv2 simulated-telemetry replay used by the local
  benchmark demonstration, pending review of upstream redistribution terms.
- Per-window outputs, waveform rows, witness rows, and audit traces.
- The unlicensed RINS-T source checkout and unused third-party baselines.
- Literature libraries, internal workflow state, superseded figures, and old
  release archives.

The public checkout consequently has no benchmark preset. The browser directs
users to compatible CSV inspection in review-only mode; it does not substitute
the hidden synthetic test fixture. These exclusions prevent accidental
redistribution of upstream data, private research traces, unlicensed
third-party source, and obsolete material. They do not imply that the retained
research archive was deleted.

The MIT License applies to the included DenoiseAPT code. Dataset and third-party
terms remain with their respective upstream sources.
