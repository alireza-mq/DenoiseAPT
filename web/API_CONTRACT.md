# DenoiseAPT web API contract

The browser client is static and calls the API on the same origin. All responses use `application/json`. Error responses should use an appropriate HTTP status and return:

```json
{"error": "A concise message safe to show to the user."}
```

Arrays representing one analysis window must have identical lengths. Numeric arrays must contain finite JSON numbers (not `NaN` or `Infinity`). Metric rates are fractions in `[0, 1]`; the client converts them to percentages for display. All interval bounds use the half-open convention `[start, end)`.

The live service uses the packaged protocol-v1 seed-17 validation-selected soft
generator, the matched ordinary cGAN, detectors A/B, and the development-frozen
automatic controller. Inference is one deterministic eval-mode forward pass
after observation-based per-window normalization. The API never reads
protocol-v2 confirmation inputs or results.

## `GET /api/health`

Used on page load. A successful response marks the analysis service ready.

```json
{
  "status": "ok",
  "device": "cpu",
  "models_ready": true,
  "max_request_bytes": 16777216
}
```

Only `status` is required. `device`, `models_ready`, and `max_request_bytes`
are optional display/diagnostic fields. The browser uses `max_request_bytes`
to reject oversized encoded requests before transmission.

## `GET /api/cases`

Returns all locally prepared catalog cases, including the clearly marked
synthetic fixture and optional benchmark cases.

```json
{
  "cases": [
    {
      "id": "smd-machine-1-1",
      "name": "SMD · machine-1-1",
      "domain": "server telemetry",
      "length": 28479,
      "sample_rate": 1.0,
      "anomaly_count": 5,
      "benchmark_case": true,
      "synthetic": false
    }
  ]
}
```

Required per case: `id`, `name`. The remaining fields are optional but improve window validation and context in the interface.

## `POST /api/analyze`

Runs controlled corruption, all configured denoisers, anomaly scoring, concern estimation, and evaluation. Exactly one of `case_id` or `upload` is sent.

Benchmark request:

```json
{
  "case_id": "smd-machine-1-1",
  "corruption": {"family": "gaussian", "severity": 0.3, "seed": 42},
  "window": {"start": 4000, "length": 512}
}
```

Uploaded signal request:

```json
{
  "upload": {
    "name": "sensor.csv",
    "values": [0.13, 0.12, 0.15],
    "timestamps": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"],
    "labels": [0, 0, 1]
  },
  "corruption": {"family": "none", "severity": 0, "seed": 42},
  "window": {"start": 0, "length": 512}
}
```

`upload.timestamps` and `upload.labels` are optional; when supplied, each must
have the same length as `upload.values`. The supported corruption families are
`gaussian`, `impulse`, `drift`, `mixed`, and `none`. Severity is in `[0, 1]`.
The browser accepts uploaded signals with 64–250,000 finite values and requests windows of 64–2,048 samples. The service should validate the full upload, then slice the requested window before inference.

Response:

```json
{
  "session_id": "7f24bf39",
  "history_depth": 0,
  "meta": {
    "case_name": "SMD · machine-1-1",
    "domain": "server telemetry",
    "sample_rate": 1.0,
    "benchmark_case": true,
    "synthetic": false,
    "soft_generator_sha256": "5d262532...",
    "controller_artifact_sha256": "5aa8bb21...",
    "corruption": {"family": "gaussian", "severity": 0.3, "seed": 42}
  },
  "time": [4000, 4001, 4002],
  "series": {
    "reference": [0.1, 0.2, 0.3],
    "observed": [0.12, 0.24, 0.29],
    "classical": [0.11, 0.21, 0.28],
    "cgan": [0.1, 0.19, 0.27],
    "denoiseapt": [0.1, 0.2, 0.29],
    "soft_candidate": [0.1, 0.2, 0.29],
    "hybrid": [0.1, 0.21, 0.28],
    "automatic": [0.1, 0.21, 0.30],
    "approved": [0.1, 0.21, 0.30]
  },
  "scores": {
    "observed": [0.02, 0.07, 0.8],
    "classical": [0.01, 0.04, 0.62],
    "cgan": [0.01, 0.03, 0.31],
    "denoiseapt": [0.01, 0.05, 0.74],
    "soft_candidate": [0.01, 0.05, 0.74],
    "hybrid": [0.01, 0.05, 0.73],
    "automatic": [0.01, 0.06, 0.77],
    "approved": [0.01, 0.06, 0.77]
  },
  "automatic_control": {
    "mode": "witness_certificate",
    "certification_eligible": true,
    "eligibility_reason": "A/B thresholds have frozen UCR-Medical validation provenance for this 512-sample contract.",
    "decision": "blend",
    "auto_committed": true,
    "fallback_reason": null,
    "beta": [0.0, 0.25, 0.0],
    "repair_intervals": [
      {"start": 1, "end": 2, "beta": 0.25, "action": "blend", "reasons": ["A_causal_mlp:retention"]}
    ],
    "certificate": {
      "status": "passed",
      "passed": true,
      "witnesses": [
        {"witness_id": "A_causal_mlp", "preservation_passed": true, "fabrication_passed": true},
        {"witness_id": "B_causal_conv", "preservation_passed": true, "fabrication_passed": true}
      ],
      "limitations": ["Certificate applies only to configured deterministic witnesses and frozen thresholds."]
    },
    "controller_latency_ms": 2.7,
    "audit": {"decision_hash": "sha256..."}
  },
  "hybrid_control": {
    "mode": "witness_certificate",
    "decision": "accept",
    "auto_committed": true,
    "fallback_reason": null,
    "denoiseapt_repair_source_kind": "denoiseapt_automatic",
    "routing_latency_ms": 1.9,
    "hybrid_latency_ms": 10.0,
    "routing": {
      "algorithm_version": "evidence-gated-classical-dapt-v1",
      "hard_routed_fraction": 0.109375,
      "nonzero_routed_fraction": 0.140625,
      "mean_denoiseapt_weight": 0.125
    },
    "certificate": {"status": "passed", "passed": true}
  },
  "concern": {
    "values": [0.08, 0.31, 0.82],
    "levels": ["low", "low", "high"]
  },
  "cues": {
    "score_change": [0.03, 0.12, 0.79],
    "morphology": [0.02, 0.18, 0.71],
    "disagreement": [0.0, 0.0, 0.0]
  },
  "anomaly_intervals": [
    {"start": 2, "end": 3, "label": "point anomaly"}
  ],
  "metrics": {
    "classical": {"rmse": 0.11, "mae": 0.08, "snr_improvement": 2.1, "vus_pr_approx": 0.61, "event_recall": 0.8, "erasure_rate": 0.2, "false_event_rate": 0.03},
    "cgan": {"rmse": 0.09, "mae": 0.07, "snr_improvement": 3.2, "vus_pr_approx": 0.49, "event_recall": 0.6, "erasure_rate": 0.4, "false_event_rate": 0.04},
    "denoiseapt": {"rmse": 0.10, "mae": 0.07, "snr_improvement": 3.0, "vus_pr_approx": 0.73, "event_recall": 1.0, "erasure_rate": 0.0, "false_event_rate": 0.02, "latency_ms": 5.4},
    "automatic": {"rmse": 0.10, "mae": 0.07, "snr_improvement": 3.0, "vus_pr_approx": 0.75, "event_recall": 1.0, "erasure_rate": 0.0, "false_event_rate": 0.02, "latency_ms": 8.1},
    "hybrid": {"rmse": 0.09, "mae": 0.06, "snr_improvement": 3.4, "vus_pr_approx": 0.74, "event_recall": 1.0, "erasure_rate": 0.0, "false_event_rate": 0.0, "latency_ms": 10.0},
    "approved": {"rmse": 0.10, "mae": 0.07, "snr_improvement": 3.0, "vus_pr_approx": 0.75, "event_recall": 1.0, "erasure_rate": 0.0, "false_event_rate": 0.02}
  }
}
```

Required fields are `session_id`, `series.observed`, `series.denoiseapt`,
`series.automatic`, `series.hybrid`, `series.approved`, `scores.observed`,
`scores.denoiseapt`, `scores.automatic`, `scores.hybrid`, `automatic_control`,
`hybrid_control`, and `concern.values`.
`series.denoiseapt` and `series.soft_candidate` are compatibility aliases for
the generative repair candidate. `series.approved` initially equals
`series.automatic`, including
when the controller abstains to the exact observation. `series.reference` and
reference-based metrics are omitted for an uploaded observation without a
clean reference. `time` defaults to local sample indices when omitted.
`anomaly_intervals` use zero-based, half-open indices relative to the returned
window.

`series.classical` is retained as a stable API key for compatibility; its
current value is the exact width-9, reflect-padded signal filter shown in the
interface as **Moving-average filter (w=9)**.

`series.hybrid` is a separately evaluated evidence-gated comparison. It
uses the exact reflect-padded moving-average result where that result satisfies the frozen
A/B witness contract, and routes failed witness support to DenoiseAPT with a
fixed tapered boundary. `hybrid_control` records the route mask, checks,
eligibility, hashes, and fallback. The established `automatic` output and the
current-session `approved` signal are intentionally unchanged. For uploads and
other out-of-scope requests the hybrid is review-only; “no witness violation”
must not be described as “no anomaly.”

Only the explicitly allowlisted 512-sample UCR-Medical packaged case has
frozen threshold-domain provenance. Uploads, synthetic cases, altered window
contracts, and unrecognized domains return:

```json
{
  "mode": "review_only",
  "certification_eligible": false,
  "decision": "review_only",
  "auto_committed": false,
  "certificate": {"status": "unverified", "passed": false}
}
```

In review-only mode the soft model output is still exposed as `automatic` and
is the initial editable baseline, but the UI must not call it certified or
automatically committed. Score-dependent event metrics are omitted because
the frozen threshold is out of scope.

`vus_pr_approx` is a transparent, lightweight approximation for interactive demonstration only. It must not be described as the official VUS-PR metric. Publication tables must be generated with the official evaluator. The browser also accepts the legacy response key `vus_pr` as a compatibility fallback, but new server implementations should return `vus_pr_approx`.

`metrics.denoiseapt.latency_ms` is the soft generator's deterministic one-pass
latency. `metrics.automatic.latency_ms` is the soft-generator plus controller
latency reported by the frozen controller. `metrics.hybrid.latency_ms` is the
end-to-end upstream DenoiseAPT plus hybrid-routing latency; the routing-only
component is `hybrid_control.routing_latency_ms`. These browser timings are useful
for demonstration diagnostics and are not substitutes for the controlled
publication timing protocol.

The client classifies concern values `<0.32` as low, `0.32–0.619…` as medium, and `≥0.62` as high, matching the packaged checkpoint. These labels are inspection levels, not probabilities.

## `POST /api/intervene`

Applies one reversible action to the current session. `start` is inclusive and `end` is exclusive; both are zero-based indices in the returned window. `beta` is the observation weight for blending:

`approved[t] = beta * observed[t] + (1 - beta) * denoiseapt[t]`.

```json
{
  "session_id": "7f24bf39",
  "action": "blend",
  "start": 173,
  "end": 196,
  "beta": 0.5
}
```

Actions:

- `accept`: use the generative repair candidate on the requested interval.
- `protect`: use observed samples in the selected interval.
- `blend`: blend the observation and DenoiseAPT candidate in the selected interval.
- `restore_automatic`: restore the immutable automatic baseline for the whole window.
- `revert`: undo the most recent session action.

Response:

```json
{
  "series": {"approved": [0.1, 0.21, 0.31]},
  "scores": {"approved": [0.01, 0.06, 0.77]},
  "metrics": {
    "approved": {"rmse": 0.1, "mae": 0.07, "snr_improvement": 2.9, "vus_pr_approx": 0.75, "event_recall": 1.0, "erasure_rate": 0.0, "false_event_rate": 0.02}
  },
  "history_depth": 1,
  "revision": 1,
  "automatic_control": {
    "mode": "witness_certificate",
    "certification_eligible": true,
    "decision": "human_override",
    "auto_committed": false,
    "current_is_automatic": false,
    "certificate": {"status": "passed", "passed": true},
    "human_intervention": {"action": "blend", "revision": 1, "actions": []}
  }
}
```

`series.approved` is required. `scores.approved`, `metrics.approved`,
`history_depth`, `revision`, and the updated witness certificate should be
returned so all linked views update. A human override that fails the frozen
witness checks is reported as `status: "overridden"`; the service does not
silently preserve the earlier passed badge. While a human edit is active,
`decision` is `human_override` and `auto_committed` is false. Restoring the
immutable baseline restores its original decision and commitment state.
`restore_automatic` is itself
reversible, so a subsequent `revert` returns to the preceding human edit.
