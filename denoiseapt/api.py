"""HTTP-facing orchestration for the local DenoiseAPT demonstration."""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .metrics import (
    anomaly_erasure_rate,
    event_recall,
    false_event_generation_rate,
    signal_metrics,
    vus_pr_approximation,
)


SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 32


class ApiError(ValueError):
    """A request error that can be safely returned to the browser."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_finite_float(v) for v in value.reshape(-1)]
    if isinstance(value, (np.floating, float)):
        return _finite_float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _score_metrics(
    reference: np.ndarray | None,
    observed: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray | None,
    before_scores: np.ndarray,
    after_scores: np.ndarray,
    latency_ms: float | None = None,
    *,
    score_threshold: float | None = None,
    threshold_source: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if reference is not None:
        result.update(_jsonable(signal_metrics(reference, candidate, noisy=observed)))
        if "snr_improvement_db" in result:
            result["snr_improvement"] = result["snr_improvement_db"]

    if labels is not None and np.any(labels) and score_threshold is not None:
        threshold = float(score_threshold)
        result["event_recall"] = _jsonable(
            event_recall(labels, after_scores, threshold)
        )
        result["erasure_rate"] = _jsonable(
            anomaly_erasure_rate(labels, before_scores, after_scores, threshold)
        )
        result["false_event_rate"] = _jsonable(
            false_event_generation_rate(labels, before_scores, after_scores, threshold)
        )
        approximation = _jsonable(vus_pr_approximation(labels, after_scores))
        if isinstance(approximation, dict):
            result["vus_pr_approx"] = approximation.get("value")
            result["vus_pr_method"] = approximation.get("method")
        else:
            result["vus_pr_approx"] = approximation
        result["score_threshold"] = threshold
        result["threshold_source"] = threshold_source
    if latency_ms is not None:
        result["latency_ms"] = float(latency_ms)
    return result


def _load_prepared_case(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Load one prepared case and enforce the shared catalog/analysis schema."""

    with np.load(path, allow_pickle=False) as item:
        signal = np.asarray(item["signal"], dtype=np.float64).reshape(-1)
        labels = (
            np.asarray(item["labels"], dtype=np.int8).reshape(-1)
            if "labels" in item.files
            else None
        )
        metadata: dict[str, Any] = {}
        if "metadata_json" in item.files:
            decoded = json.loads(str(item["metadata_json"].item()))
            if not isinstance(decoded, dict):
                raise ValueError("metadata_json must decode to an object")
            metadata = decoded
    if not np.isfinite(signal).all():
        raise ValueError("prepared signal contains NaN or infinity")
    if len(signal) < 64:
        raise ValueError("prepared signal must contain at least 64 samples")
    if np.std(signal) < 1e-10:
        raise ValueError("prepared signal must vary")
    if labels is not None and len(labels) != len(signal):
        raise ValueError("prepared labels and signal must have equal length")
    return signal, labels, metadata


class DemoService:
    """Coordinates data, model inference, metrics, and reversible sessions."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.config = json.loads((self.root / "config" / "demo.json").read_text("utf-8"))
        self._runtime = None
        self._runtime_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        # Intervention state is deliberately process-local and bounded.  It is
        # not durable storage and is never part of the scientific run record.
        self._sessions: dict[str, dict[str, Any]] = {}
        self._sessions_lock = threading.Lock()
        self.started_at = time.time()

    @property
    def checkpoint_path(self) -> Path:
        return (
            self.root
            / "checkpoints"
            / "automatic_preservation"
            / "runtime_manifest.json"
        )

    def _load_runtime(self):
        if self._runtime is None:
            with self._runtime_lock:
                if self._runtime is None:
                    if not self.checkpoint_path.exists():
                        raise ApiError(
                            "Automatic-preservation runtime is missing. Run "
                            "scripts/package_automatic_runtime.py before starting the demo."
                        )
                    from denoiseapt.automatic_runtime import AutomaticPreservationRuntime

                    self._runtime = AutomaticPreservationRuntime(self.root)
        return self._runtime

    def _runtime_status(self) -> dict[str, Any]:
        """Report core inference and optional certificate readiness separately."""

        try:
            runtime = self._load_runtime()
        except Exception as exc:  # Health must expose, rather than mask, load failure.
            return {
                "automatic_runtime_ready": False,
                "automatic_runtime_error": str(exc),
                "guided_certificate_ready": False,
                "guided_certificate_reason": "Core automatic runtime is unavailable.",
            }
        guided = runtime.guided_certificate_status
        return {
            "automatic_runtime_ready": bool(runtime.ready),
            "automatic_runtime_error": None,
            "guided_certificate_ready": bool(guided["ready"]),
            "guided_certificate_reason": str(guided["reason"]),
        }

    def health(self) -> dict[str, Any]:
        runtime_status = self._runtime_status()
        guided_record = (
            self.root / "data" / "prepared" / "tsb_ad_ucr_medical_guided.npz"
        )
        return {
            "status": "ok",
            "service": "DenoiseAPT",
            "version": __version__,
            "device": "cpu",
            "max_request_bytes": int(self.config["server"]["max_request_bytes"]),
            "model_ready": runtime_status["automatic_runtime_ready"],
            "models_ready": runtime_status["automatic_runtime_ready"],
            "automatic_runtime_ready": runtime_status["automatic_runtime_ready"],
            "automatic_runtime_error": runtime_status["automatic_runtime_error"],
            "automatic_runtime_manifest": str(self.checkpoint_path.relative_to(self.root)),
            "guided_case_ready": guided_record.is_file(),
            "guided_certificate_ready": runtime_status["guided_certificate_ready"],
            "guided_certificate_reason": runtime_status["guided_certificate_reason"],
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }

    def list_cases(self) -> dict[str, Any]:
        runtime_status = self._runtime_status()
        guided_certificate_ready = bool(
            runtime_status.get("guided_certificate_ready", False)
        )
        cases = []
        warnings = []
        prepared = self.root / "data" / "prepared"
        for path in sorted(prepared.glob("*.npz")):
            try:
                signal, labels, metadata = _load_prepared_case(path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                # One damaged optional case must not disable the packaged
                # synthetic review-only workflow.
                warnings.append(
                    {
                        "case_id": path.stem,
                        "error": f"Prepared case was ignored: {exc}",
                    }
                )
                continue
            cases.append(
                {
                    "id": path.stem,
                    "name": metadata.get("name", path.stem.replace("_", " ").title()),
                    "domain": metadata.get(
                        "domain",
                        "Synthetic" if "synthetic" in path.stem else "Unknown",
                    ),
                    "length": len(signal),
                    "sample_rate": metadata.get("sample_rate"),
                    "anomaly_count": len(_intervals(labels)),
                    "source": metadata.get("source", "packaged fixture"),
                    "benchmark_case": bool(metadata.get("benchmark_case", False)),
                    "synthetic": bool(metadata.get("synthetic", False)),
                    "automatic_certificate_available": (
                        path.stem == "tsb_ad_ucr_medical_guided"
                        and guided_certificate_ready
                    ),
                }
            )
        return {"cases": cases, "warnings": warnings}

    def _read_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        safe_id = Path(case_id).name
        if safe_id != case_id or not safe_id:
            raise ApiError("Invalid case identifier.")
        path = self.root / "data" / "prepared" / f"{safe_id}.npz"
        if not path.exists():
            raise ApiError(f"Unknown case: {case_id}")
        try:
            return _load_prepared_case(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ApiError(f"Prepared case is invalid: {case_id}") from exc

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ApiError("The request body must be a JSON object.")
        upload = payload.get("upload")
        is_upload = upload is not None
        if is_upload and payload.get("case_id"):
            raise ApiError("Provide either upload or case_id, not both.")
        if is_upload:
            if not isinstance(upload, dict):
                raise ApiError("upload must be a JSON object.")
            values = upload.get("values")
            if not isinstance(values, list):
                raise ApiError("Uploaded CSV values are missing.")
            reference = _validated_signal(values)
            if not 64 <= len(reference) <= 250_000:
                raise ApiError("Upload must contain between 64 and 250,000 samples.")
            labels_raw = upload.get("labels")
            labels = (
                _validated_labels(labels_raw, len(reference))
                if labels_raw is not None
                else None
            )
            timestamps = upload.get("timestamps")
            if timestamps is not None:
                if not isinstance(timestamps, list) or len(timestamps) != len(reference):
                    raise ApiError("Timestamps and signal values must have equal length.")
                timestamps = [str(value) for value in timestamps]
            metadata = {
                "name": str(upload.get("name") or "Uploaded CSV"),
                "domain": "Uploaded",
                "source": "local browser upload",
                "uploaded": True,
                "benchmark_case": False,
                "synthetic": False,
            }
            case_id = "upload"
        else:
            case_id = str(payload.get("case_id") or "synthetic_guided_case")
            reference, labels, metadata = self._read_case(case_id)
            timestamps = None

        window = payload.get("window")
        if window is None:
            window = {}
        elif not isinstance(window, dict):
            raise ApiError("window must be a JSON object.")
        start = int(window.get("start", 0))
        length = int(window.get("length", min(len(reference), self.config["window_length"])))
        length = max(64, min(length, 2048, len(reference)))
        start = max(0, min(start, len(reference) - length))
        reference = reference[start : start + length]
        if labels is not None:
            labels = labels[start : start + length]
        if timestamps:
            timestamps = timestamps[start : start + length]

        corruption_request = payload.get("corruption")
        if corruption_request is None:
            corruption_request = self.config["corruption"]
        elif not isinstance(corruption_request, dict):
            raise ApiError("corruption must be a JSON object.")
        family = str(corruption_request.get("family", "mixed"))
        severity = float(corruption_request.get("severity", 0.32))
        seed = int(corruption_request.get("seed", self.config["seed"]))
        if not 0 <= severity <= 1:
            raise ApiError("Corruption severity must be between 0 and 1.")
        from denoiseapt.corruptions import apply_measurement_corruption

        corruption = apply_measurement_corruption(reference, family, severity, seed)
        observed = np.asarray(corruption.corrupted, dtype=np.float64)
        metric_reference = reference
        if is_upload and (family == "none" or severity == 0):
            # With no controlled corruption, uploaded samples are the observed
            # signal y; a clean counterpart x is unavailable.
            metric_reference = None

        runtime = self._load_runtime()
        with self._inference_lock:
            inference = runtime.run(
                observed,
                case_id=case_id,
                metadata=metadata,
                is_upload=is_upload,
            )
            hybrid_inference = runtime.run_hybrid(inference)
        candidate = np.asarray(inference.soft_candidate, dtype=np.float64)
        baseline = np.asarray(inference.ordinary_candidate, dtype=np.float64)
        automatic = np.asarray(inference.automatic, dtype=np.float64)
        hybrid = np.asarray(hybrid_inference.output, dtype=np.float64)
        score_map = inference.scores["A_causal_mlp"]
        observed_scores = np.asarray(score_map["observation"], dtype=np.float64)
        candidate_scores = np.asarray(score_map["soft_candidate"], dtype=np.float64)
        baseline_scores = np.asarray(score_map["ordinary_cgan"], dtype=np.float64)
        automatic_scores = np.asarray(score_map["automatic"], dtype=np.float64)
        # ``classical`` is a stable legacy response key.  Its value is the exact
        # float32, reflect-padded moving-average candidate checked by the router,
        # so the displayed signal, metrics, and witness scores stay aligned.
        classical = np.asarray(hybrid_inference.classical_candidate, dtype=np.float64)
        classical_reports = {
            item.witness_id: item.output_scores
            for item in hybrid_inference.routing.classical_verification.witnesses
        }
        classical_score_values = classical_reports.get("A_causal_mlp")
        if classical_score_values is None:
            # Never invent a zero anomaly-score timeline after a transient
            # witness failure.  A fresh score either succeeds or the request
            # fails closed instead of returning mismatched signal/metrics.
            classical_score_values = runtime.score_signal(
                classical, inference.normalization, witness_id="A_causal_mlp"
            )
        classical_scores = np.asarray(
            classical_score_values,
            dtype=np.float64,
        )
        hybrid_scores = np.asarray(
            hybrid_inference.scores["A_causal_mlp"], dtype=np.float64
        )

        session_id = uuid.uuid4().hex
        session = inference.new_session()
        created = time.time()
        session_data = {
            "created": created,
            "model_session": session,
            "inference": inference,
            "reference": metric_reference,
            "labels": labels,
            "observed": observed,
            "observed_scores": observed_scores,
        }
        with self._sessions_lock:
            self._prune_sessions(created)
            self._sessions[session_id] = session_data

        threshold = (
            float(runtime.thresholds["A_causal_mlp"])
            if inference.eligibility.eligible
            else None
        )
        threshold_source = (
            str(runtime.threshold_scope.get("threshold_source"))
            if threshold is not None
            else None
        )
        metric_kwargs = {
            "score_threshold": threshold,
            "threshold_source": threshold_source,
        }
        metrics = {
            "classical": _score_metrics(
                metric_reference,
                observed,
                classical,
                labels,
                observed_scores,
                classical_scores,
                **metric_kwargs,
            ),
            "cgan": _score_metrics(
                metric_reference,
                observed,
                baseline,
                labels,
                observed_scores,
                baseline_scores,
                latency_ms=float(inference.ordinary_generator_latency_ms),
                **metric_kwargs,
            ),
            "denoiseapt": _score_metrics(
                metric_reference,
                observed,
                candidate,
                labels,
                observed_scores,
                candidate_scores,
                float(inference.soft_generator_latency_ms),
                **metric_kwargs,
            ),
            "automatic": _score_metrics(
                metric_reference,
                observed,
                automatic,
                labels,
                observed_scores,
                automatic_scores,
                (
                    None
                    if inference.projection is None
                    else float(inference.projection.total_latency_ms or 0.0)
                ),
                **metric_kwargs,
            ),
            "hybrid": _score_metrics(
                metric_reference,
                observed,
                hybrid,
                labels,
                observed_scores,
                hybrid_scores,
                float(hybrid_inference.hybrid_latency_ms),
                **metric_kwargs,
            ),
        }
        metrics["approved"] = dict(metrics["automatic"])
        automatic_control = inference.automatic_payload()
        response = {
            "session_id": session_id,
            "history_depth": 0,
            "revision": 0,
            "meta": {
                "case_id": case_id,
                "case_name": metadata.get("name", case_id),
                "domain": metadata.get("domain", "Unknown"),
                "source": metadata.get("source"),
                "sample_rate": metadata.get("sample_rate"),
                "window_start": start,
                "corruption": _jsonable(corruption.metadata()),
                "model": "protocol-v1 seed-17 preservation-trained generative repair network",
                "selected_model": runtime.selected_model,
                "soft_generator_sha256": inference.runtime_provenance.get(
                    "soft_generator_sha256"
                ),
                "ordinary_generator_sha256": inference.runtime_provenance.get(
                    "ordinary_generator_sha256"
                ),
                "detector_checkpoint_sha256": inference.runtime_provenance.get(
                    "detector_checkpoint_sha256"
                ),
                "controller_artifact_sha256": inference.runtime_provenance.get(
                    "controller_artifact_sha256"
                ),
                "hybrid_algorithm_version": hybrid_inference.runtime_provenance.get(
                    "hybrid_algorithm_version"
                ),
                "hybrid_config_sha256": hybrid_inference.runtime_provenance.get(
                    "hybrid_config_sha256"
                ),
                "inference_rule": (
                    "one deterministic eval-mode forward pass after per-window "
                    "observation normalization"
                ),
                "reference_available": metric_reference is not None,
                "benchmark_case": bool(metadata.get("benchmark_case", False)),
                "synthetic": bool(metadata.get("synthetic", False)),
                "review_only": not inference.eligibility.eligible,
            },
            "time": timestamps if timestamps else list(range(start, start + length)),
            "series": {
                "observed": observed,
                "classical": classical,
                "cgan": baseline,
                "denoiseapt": candidate,
                "soft_candidate": candidate,
                "automatic": automatic,
                "hybrid": hybrid,
                "approved": automatic,
            },
            "scores": {
                "observed": observed_scores,
                "classical": classical_scores,
                "cgan": baseline_scores,
                "denoiseapt": candidate_scores,
                "soft_candidate": candidate_scores,
                "automatic": automatic_scores,
                "hybrid": hybrid_scores,
                "approved": automatic_scores,
            },
            "automatic_control": automatic_control,
            "hybrid_control": hybrid_inference.control_payload(),
            "concern": _concern_response(inference.concern.to_dict(), length),
            "cues": _cues_response(inference.concern.to_dict(), length),
            "anomaly_intervals": _intervals(labels) if labels is not None else [],
            "metrics": metrics,
            "limitations": [
                "Concern values are inspection cues, not calibrated failure probabilities.",
                "A passed certificate applies only to the frozen A/B witnesses and thresholds.",
                (
                    "The certificate does not establish physical anomaly truth or "
                    "unseen-detector transfer."
                ),
                (
                    "The controller configuration is development-frozen; "
                    "confirmation effectiveness is not claimed here."
                ),
                (
                    "The hybrid is a separately evaluated, witness-gated comparison "
                    "and is not the established live default."
                ),
            ],
        }
        if not inference.eligibility.eligible:
            response["limitations"].insert(
                0,
                "Review-only mode: no calibrated threshold provenance applies to this request.",
            )
        if metric_reference is not None:
            response["series"]["reference"] = metric_reference
        return _jsonable(response)

    def intervene(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "")
        with self._sessions_lock:
            data = self._sessions.get(session_id)
        if data is None:
            raise ApiError("Unknown or expired analysis session.")
        action = str(payload.get("action") or "").lower()
        if action not in {"accept", "protect", "blend", "restore_automatic", "revert"}:
            raise ApiError(
                "Action must be accept, protect, blend, restore_automatic, or revert."
            )
        model_session = data["model_session"]
        runtime = self._load_runtime()
        with self._inference_lock:
            if action in {"revert", "restore_automatic"}:
                model_session.apply(
                    action, expected_revision=payload.get("expected_revision")
                )
            else:
                n = len(data["observed"])
                start = max(0, min(int(payload.get("start", 0)), n - 1))
                end = max(start + 1, min(int(payload.get("end", n)), n))
                beta = float(payload.get("beta", 0.5))
                if not 0 <= beta <= 1:
                    raise ApiError("Blend coefficient must be between 0 and 1.")
                model_session.apply(
                    action,
                    start,
                    end,
                    beta=beta,
                    expected_revision=payload.get("expected_revision"),
                )
            session_dict = model_session.to_dict()
            approved = np.asarray(session_dict.get("current"), dtype=np.float64)
            scores = np.asarray(
                runtime.score_signal(approved, data["inference"].normalization),
                dtype=np.float64,
            )
        threshold = (
            float(runtime.thresholds["A_causal_mlp"])
            if data["inference"].eligibility.eligible
            else None
        )
        metrics = _score_metrics(
            data["reference"],
            data["observed"],
            approved,
            data["labels"],
            data["observed_scores"],
            scores,
            score_threshold=threshold,
            threshold_source=(
                str(runtime.threshold_scope.get("threshold_source"))
                if threshold is not None
                else None
            ),
        )
        current_is_automatic = bool(
            np.array_equal(approved.astype(np.float32), data["inference"].automatic)
        )
        baseline_control = data["inference"].automatic_payload()
        control_update = {
            "mode": session_dict.get("mode"),
            "certification_eligible": bool(
                data["inference"].eligibility.eligible
            ),
            "eligibility_reason": data["inference"].eligibility.reason,
            "decision": (
                baseline_control.get("decision")
                if current_is_automatic
                else "human_override"
            ),
            "auto_committed": bool(
                current_is_automatic and baseline_control.get("auto_committed") is True
            ),
            "fallback_reason": (
                baseline_control.get("fallback_reason") if current_is_automatic else None
            ),
            "certificate": session_dict.get("certificate"),
            "current_is_automatic": current_is_automatic,
            "human_intervention": {
                "action": action,
                "revision": int(session_dict.get("revision", 0)),
                "actions": session_dict.get("actions", []),
            },
        }
        return _jsonable(
            {
                "session_id": session_id,
                "series": {"approved": approved},
                "scores": {"approved": scores},
                "metrics": {"approved": metrics},
                "history_depth": int(session_dict.get("history_depth", 0)),
                "revision": int(session_dict.get("revision", 0)),
                "action": action,
                "automatic_control": control_update,
            }
        )

    def _prune_sessions(self, now: float) -> None:
        expired = [
            key
            for key, value in self._sessions.items()
            if now - value["created"] > SESSION_TTL_SECONDS
        ]
        for key in expired:
            self._sessions.pop(key, None)
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda k: self._sessions[k]["created"])
            self._sessions.pop(oldest, None)


def _validated_signal(values: list[Any]) -> np.ndarray:
    try:
        signal = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ApiError("Signal values must be numeric.") from exc
    if not np.isfinite(signal).all():
        raise ApiError("Signal values cannot contain NaN or infinity.")
    if np.std(signal) < 1e-10:
        raise ApiError("Signal is constant; a varying time series is required.")
    return signal


def _validated_labels(values: list[Any], length: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(labels) != length:
        raise ApiError("Label and signal columns must have equal length.")
    if not np.isfinite(labels).all():
        raise ApiError("Labels cannot contain NaN or infinity.")
    return (labels > 0).astype(np.int8)


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    """Compatibility oracle for frozen width-9 moving-average (MA9) parity tests."""

    width = max(3, int(width) | 1)
    pad = width // 2
    padded = np.pad(values, pad, mode="reflect")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")


def _intervals(labels: np.ndarray | None) -> list[dict[str, Any]]:
    if labels is None or len(labels) == 0:
        return []
    binary = (np.asarray(labels) > 0).astype(np.int8)
    changes = np.diff(np.r_[0, binary, 0])
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        {"start": int(start), "end": int(end), "label": "labeled anomaly"}
        for start, end in zip(starts, ends)
    ]


def _array_or_zeros(mapping: dict[str, Any], keys: tuple[str, ...], length: int) -> np.ndarray:
    for key in keys:
        if key in mapping:
            arr = np.asarray(mapping[key], dtype=np.float64).reshape(-1)
            if len(arr) == length:
                return arr
    return np.zeros(length, dtype=np.float64)


def _concern_response(concern: dict[str, Any], length: int) -> dict[str, Any]:
    values = _array_or_zeros(concern, ("concern", "values", "risk"), length)
    level = concern.get("level")
    if not isinstance(level, list) or len(level) != length:
        level = [
            "high" if value >= 0.66 else "medium" if value >= 0.33 else "low"
            for value in values
        ]
    return {
        "values": values,
        "levels": level,
        "intervals": concern.get("intervals", []),
        "interpretation": concern.get(
            "interpretation", "Inspection priority; not a calibrated probability."
        ),
    }


def _cues_response(concern: dict[str, Any], length: int) -> dict[str, Any]:
    return {
        "score_change": _array_or_zeros(
            concern, ("score_delta", "score_change", "possible_suppression"), length
        ),
        "morphology": _array_or_zeros(concern, ("morphology",), length),
        "disagreement": _array_or_zeros(concern, ("uncertainty", "disagreement"), length),
    }
