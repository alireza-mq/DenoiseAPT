# Data card

## Intended data paths

DenoiseAPT exposes two user-facing paths with different scientific scopes:

1. two integrity-pinned benchmark replays for the guided evaluation
   demonstration; and
2. compatible local CSV inspection in explicit review-only mode.

The deterministic fixture bundled with the public code is reserved for API and
installation tests. It is hidden from the revised benchmark selector.

## Primary benchmark source

The replays are derived from the univariate TSB-AD benchmark:

- Project: <https://github.com/TheDatumOrg/TSB-AD>
- Official archive: <https://www.thedatum.org/datasets/TSB-AD-U.zip>
- Dataset and license summary:
  <https://thedatumorg.github.io/TSB-AD/#summary-of-datasets>
- Paper: Qinghua Liu and John Paparrizos, *The Elephant in the Room:
  Towards a Reliable Time-Series Anomaly Detection Benchmark*, NeurIPS 2024
  Datasets and Benchmarks Track.

## CATSv2 main-workflow replay

The guided demonstration uses a fixed 512-point CATSv2 simulated-telemetry
window from the TSB-AD-U Sensor confirmation panel. Its source group is held out
from fitting, development, and calibration. The replay binds the source window,
evaluation-only reference and labels, one controlled corruption condition, and
the frozen matched output of every displayed denoiser to an integrity manifest.

CATSv2 represents commands, stimuli, and telemetry from a simulated complex
dynamical system with injected anomalies. The TSB-AD dataset summary lists it
under CC BY 4.0. It is not presented as newly collected or real-world sensor
data.

The displayed condition was selected after the benchmark panel was inspected
because its repeated changes make the interface workflow easier to see. It is
an illustrative replay, not an additional benchmark experiment, and aggregate
claims must remain tied to the registered aggregate evaluation.

The replay is bundled with an integrity manifest and is governed by CC BY 4.0,
as recorded in `LICENSES/CATS-DATA-NOTICE.md`. It is not covered by the project
MIT license.

## MIT-BIH ECG anomaly-preservation replay

The second replay uses a fixed 512-time-point ECG window from the MIT-BIH
Arrhythmia Database constituent of the held-out TSB-AD-U Medical confirmation
panel. The clean reference and corrupted observation both contain configured
Scorer A and B event opportunities near time index 232. Our Model retains both;
Median Filter and RINS-T lose both, Wavelet loses one, and Noisereduce retains
both but has higher reconstruction error on this condition.

This replay was also selected after panel inspection and is illustrative. The
configured evidence is not a medical diagnosis, and the interface makes no
claim about unseen scorers. The artifact is governed by ODC Attribution 1.0,
with attribution and citation in `LICENSES/MITDB-DATA-NOTICE.md`; it is not
covered by the project MIT license.

## Frozen comparison outputs

For either replay, the reference, corrupted observation, Median Filter,
Wavelet Thresholding, Noisereduce, RINS-T Adaptation, and Our Model traces all
belong to the same frozen condition. External comparators are not executed when
the browser request is made. The reference and labels support evaluation views
only and never guide routing or ordinary inference.

## Uploaded data

CSV upload is processed in memory by the local service. The accepted form is a
headered, comma-delimited, univariate numeric series. Conventional time columns
are excluded when selecting the signal column. A numeric column named `label`,
`anomaly`, or `target` (case-insensitive), when present, is used only for
display.

Uploaded values are treated as the observation. No clean pre-corruption
reference is inferred, so reference-based reconstruction measures are omitted.
The upload path supplies no resampling, missing-value imputation, multivariate
modeling, domain adaptation, or calibration for unseen domains. It remains
review-only even when labels are supplied and cannot receive held-out replay or
benchmark-certificate status.

## Hidden synthetic fixture

The public package includes a deterministic author-created fixture for
self-contained API, setup, and UI regression tests. The browser filters it out
of the benchmark selector. It is not a TSB-AD case, a supported deployment
domain, or evidence for the reported benchmark.

## Claim boundary

A configured Scorer-A/B pass is limited to the frozen witnesses, thresholds,
normalization, and eligible replay contract. It is not a calibrated anomaly
probability, a statement of physical anomaly truth, or evidence of
generalization to an uploaded or unseen domain.
