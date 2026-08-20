# DenoiseAPT web API contract

The static browser client calls the API on the same origin. Responses use
`application/json`; errors use an appropriate HTTP status and a concise message
safe to display:

```json
{"error": "A concise user-facing message."}
```

Arrays for one time window must have equal lengths and contain only finite JSON
numbers. All intervals are zero-based and half-open: `[start, end)`.

## Public data boundary

The revised browser recognizes two modes:

- two bundled, integrity-pinned benchmark replays, plus support for an optional
  integrity-matched local MSL replay; and
- local CSV inspection in explicit review-only mode.

The main replay is a derived CATSv2 simulated telemetry window. A local MSL
artifact, when installed, is ordered second as the anomaly-preservation
spotlight. The bundled MIT-BIH ECG window is ordered third. The bundled data
use separate upstream terms recorded in `LICENSES/`; neither artifact is
covered by the project MIT license.

The deterministic synthetic fixture retained in the public package is for API,
setup, and regression tests only. It may be returned by the cases endpoint, but
the browser filters it out because it is not marked as a benchmark replay.

## Revised comparison schema

A replay exposes only the following synchronized traces:

| API key | Interface label |
|---|---|
| `reference` | Reference before corruption (evaluation only) |
| `observed` | Corrupted Observation |
| `median` | Median Filter (w=3) |
| `wavelet` | Wavelet Thresholding |
| `noisereduce` | Noisereduce |
| `rins_t` | RINS-T Adaptation |
| `our_model` | Our Model |

The anomaly-evidence view uses only `scores.observed` and
`scores.our_model`. Internal generator and controller variants are not separate
comparison methods. The browser accepts `series.automatic` or
`series.denoiseapt` as a compatibility fallback for `series.our_model` on
review-only responses.

## `GET /api/health`

A successful response marks the local analysis service ready:

```json
{
  "status": "ok",
  "device": "cpu",
  "models_ready": true,
  "heldout_replay_ready": true,
  "heldout_replay_count": 2,
  "max_request_bytes": 16777216
}
```

Only `status` is required. The remaining fields are optional diagnostics.
`heldout_replay_ready` is true only when at least one replay, its manifest, and
the pinned threshold record are present and pass integrity validation.

## `GET /api/cases`

Returns locally installed prepared cases. The revised browser lists only cases
whose `benchmark_replay` field is `true`.

```json
{
  "cases": [
    {
      "id": "tsb_ad_cats_heldout_replay",
      "name": "CATS rich dynamics",
      "domain": "Sensor",
      "length": 512,
      "benchmark_replay": true,
      "held_out": true,
      "synthetic": true,
      "fixed_window": true,
      "demo_role": "main_workflow",
      "demo_order": 0,
      "default_family": "gaussian",
      "default_severity": 0.75,
      "default_replicate": 0
    }
  ],
  "default_case_id": "tsb_ad_cats_heldout_replay",
  "warnings": []
}
```

`id` and `name` are required for any case. Replay metadata improves validation
and scientific context. `demo_order` controls selector order, while
`default_case_id` identifies the main workflow. A replay must not be listed
unless its artifact and manifest pass their pinned integrity checks.

## `POST /api/analyze`

Exactly one of `case_id` or `upload` is supplied.

### Replay request

```json
{
  "case_id": "tsb_ad_cats_heldout_replay",
  "window": {"start": 0, "length": 512},
  "corruption": {"family": "gaussian", "severity": 0.75, "replicate": 0}
}
```

The replay uses its fixed time window and accepts only conditions stored in its
frozen grid. Controlled corruption reproduces an evaluation condition; it is
not an ordinary inference stage.

### CSV request

```json
{
  "upload": {
    "name": "sensor.csv",
    "values": [0.13, 0.12, 0.15],
    "timestamps": ["t0", "t1", "t2"],
    "labels": [0, 0, 1]
  },
  "window": {"start": 0, "length": 512},
  "corruption": {"family": "none", "severity": 0}
}
```

The short arrays above illustrate field alignment rather than the minimum
accepted input size. `timestamps` and `labels` are optional and, when present,
must align with `values`. The browser accepts 64--250,000 finite values and
requests one contiguous interval of 64--2,048 points. Uploads default to no
controlled corruption.

### Replay response

The replay response uses the narrow comparison roster and records its
scientific scope:

```json
{
  "session_id": "session-id",
  "history_depth": 0,
  "revision": 0,
  "meta": {
    "benchmark_replay": true,
    "held_out": true,
    "synthetic": true,
    "posthoc_visual_selection": true,
    "reference_available": true,
    "reference_scope": "pre-corruption source window; evaluation only",
    "method_scope": "frozen matched outputs",
    "display_witness": "A_causal_mlp",
    "display_witness_label": "Scorer A",
    "corruption": {"demonstration_only": true}
  },
  "time": [],
  "series": {
    "reference": [],
    "observed": [],
    "median": [],
    "wavelet": [],
    "noisereduce": [],
    "rins_t": [],
    "our_model": []
  },
  "scores": {"observed": [], "our_model": []},
  "metrics": {},
  "concern": {"values": [], "summary": {}},
  "automatic_control": {
    "mode": "heldout_benchmark_replay",
    "certification_eligible": true,
    "current_is_automatic": true,
    "certificate": {"status": "passed", "passed": true}
  },
  "limitations": []
}
```

For each displayed trace with a known reference, `metrics` may contain finite
`overall_os_nrmse` and `anomaly_os_nrmse` fields. These legacy API keys hold the
study-defined whole-window and event-region NRMSE values; `OS-NRMSE` is not
presented as a standard scientific term. This contract deliberately does not
embed condition-level results. The reference and labels are evaluation-only.
External comparator arrays are frozen matched outputs rather than browser-time
executions.

`limitations` must disclose that the condition is illustrative and selected
after panel inspection, aggregate conclusions come from the sealed benchmark,
the reference and labels are evaluation-only, and configured scorer evidence
does not establish physical anomaly truth.

### CSV review-only response

An ordinary upload is the observation, so no clean reference is inferred. The
response omits `series.reference`, matched baseline traces, and reference-based
metrics. It returns the observation and a model response for inspection, with a
review-only control record:

```json
{
  "series": {"observed": [], "automatic": [], "approved": []},
  "scores": {"observed": [], "automatic": [], "approved": []},
  "automatic_control": {
    "mode": "review_only",
    "certification_eligible": false,
    "decision": "review_only",
    "auto_committed": false,
    "certificate": {"status": "unverified", "passed": false}
  }
}
```

Supplying labels does not change this status. Uploads receive no held-out
designation, cross-domain calibration, resampling, missing-value imputation, or
multivariate support. The runtime retains additional legacy comparison keys for
API compatibility; the revised browser presents `automatic`/`approved` as the
single Our Model trace rather than treating those internal keys as separate
methods.

## `POST /api/intervene`

The revised interaction changes the current session output without retraining
the model. For a selected interval, `beta` is the observation weight:

`adapted[t] = beta * observed[t] + (1 - beta) * automatic[t]`.

```json
{
  "session_id": "session-id",
  "action": "blend",
  "start": 128,
  "end": 160,
  "beta": 0.75,
  "expected_revision": 0
}
```

Supported revised actions are:

- `blend`: apply the observation weight only inside the selected interval;
- `restore_automatic`: restore the immutable automatic baseline; and
- `revert`: undo the most recent session action.

The response returns the updated `series.our_model` (or the compatibility key
`series.approved`), current scores and available metrics, `history_depth`,
`revision`, the action record, and a fresh full-window A/B status. During an
expert edit, `current_is_automatic` is false and `auto_committed` is false.
Restoring the automatic baseline restores its original state, and restoration
itself remains reversible.

## Evidence semantics

For a replay, the local evidence band marks shared support from the display
witness named in `meta.display_witness`; the complete-window status still uses
configured Scorers A and B. For a CSV upload, the band is instead a continuous
review-only comparison cue and is not calibrated preservation evidence. Neither
display is a probability or diagnosis. A passed A/B status is limited to the
configured frozen witnesses, domain thresholds, normalization, and eligible
replay contract.
Observation-supported evidence may itself reflect measurement corruption, and
blending observation values can reintroduce that corruption. No status may be
described as proof of anomaly truth, deployment safety, or generalization to an
unseen domain.
