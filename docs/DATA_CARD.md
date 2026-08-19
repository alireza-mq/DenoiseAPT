# Data card

## Primary benchmark source

The guided case is selected from the univariate TSB-AD benchmark:

- Project: <https://github.com/TheDatumOrg/TSB-AD>
- Official archive: <https://www.thedatum.org/datasets/TSB-AD-U.zip>
- Paper: Qinghua Liu and John Paparrizos, *The Elephant in the Room:
  Towards a Reliable Time-Series Anomaly Detection Benchmark*, NeurIPS 2024
  Datasets and Benchmarks Track.

The full archive is not redistributed in the release ZIP. The downloader stores
it under `data/raw`, validates that it is a ZIP, blocks unsafe extraction paths,
and extracts only the selected member unless the user explicitly requests more.
The dataset project asks users to consult the original source licence for each
constituent dataset. The selected member's provenance is retained in the demo
case metadata.

## Offline fixture

The package also includes a deterministic synthetic fixture for installation
and API smoke tests. It is clearly identified as synthetic in the interface and
must not be reported as a TSB-AD result.

## Uploaded data

CSV upload is processed in memory by the local server. Conventional time and
label columns are excluded when selecting the best-populated numeric signal
column. A column named `Label`, `Anomaly`, or `Target` (case-insensitive), when
present, is treated as a binary anomaly label. Uploaded observations do not have
a clean reference unless one is explicitly supplied by a controlled workflow;
reference-only metrics are therefore unavailable.
