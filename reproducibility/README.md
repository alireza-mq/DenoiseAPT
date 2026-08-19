# Reproducibility material

This directory contains aggregate provenance for the retrospective matched
denoiser benchmark.

`external_denoiser_benchmark_v1/` includes the executed runner and recovery
wrapper, frozen configuration and dependency identity, development-selection
record, pre-freeze and post-recovery receipts, tests, and the aggregate benchmark
summary.

It intentionally omits raw or derived benchmark windows, condition-level
outputs, score rows, and audit traces. Those files include redistributed or
derived upstream data and are retained only in the private research archive.

The RINS-T comparison used the official architecture at the pinned commit with
the repository demo recipe and a benchmark-owned observation wrapper. Because
the pinned upstream checkout contained no software license, that third-party
source is not included. Obtain it directly from its authors if its license and
terms permit your intended use.

The included receipts establish the identity of the executed local experiment;
they do not make this reduced public snapshot independently executable without
the omitted data and third-party dependencies.
