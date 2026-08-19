"""Deterministic anomaly-evidence projection for protocol-v2 development.

The controller is deliberately not a learned anomaly detector.  It projects a
frozen denoiser candidate onto constraints induced by one or more frozen
detector witnesses.  Its certificate is consequently limited to those
witnesses, thresholds, and the recorded normalization; it is not evidence that
an event is physically genuine.

The module has no dependency on the live demo API and does not read labels.
This makes it suitable for a separately sealed confirmation runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


Decision = Literal["accept", "blend", "protect", "mixed", "abstain"]
CertificateStatus = Literal["passed", "failed", "unverified", "overridden"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: ArrayLike) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}|{array.shape}".encode("utf-8")
    return _sha256_bytes(header + b"\0" + array.tobytes())


def _signal(value: ArrayLike, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _runs(mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def _dilate(intervals: Sequence[tuple[int, int]], radius: int, size: int) -> NDArray[np.bool_]:
    result = np.zeros(size, dtype=bool)
    for start, end in intervals:
        result[max(0, start - radius) : min(size, end + radius)] = True
    return result


def _merge(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not intervals:
        return ()
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


@dataclass(frozen=True)
class WitnessSpec:
    """One deterministic detector witness and its frozen provenance."""

    witness_id: str
    scorer: Callable[[NDArray[np.float32]], ArrayLike] = field(repr=False, compare=False)
    threshold: float
    model_sha256: str
    threshold_source: str
    threshold_source_sha256: str
    context_left: int = 0
    context_right: int = 0

    def __post_init__(self) -> None:
        if not self.witness_id.strip():
            raise ValueError("witness_id cannot be empty.")
        if not callable(self.scorer):
            raise TypeError("scorer must be callable.")
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite.")
        if self.context_left < 0 or self.context_right < 0:
            raise ValueError("Witness contexts must be non-negative.")
        for name, value in (
            ("model_sha256", self.model_sha256),
            ("threshold_source", self.threshold_source),
            ("threshold_source_sha256", self.threshold_source_sha256),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty.")

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "threshold": float(self.threshold),
            "model_sha256": self.model_sha256,
            "threshold_source": self.threshold_source,
            "threshold_source_sha256": self.threshold_source_sha256,
            "context_left": int(self.context_left),
            "context_right": int(self.context_right),
        }


@dataclass(frozen=True)
class ControllerConfig:
    """Frozen finite-grid controller configuration.

    ``fabrication_dilation`` is a timing-matching tolerance for output evidence.
    It never expands the support used by the preservation endpoint: retention is
    checked on the exact original observation threshold component.
    """

    algorithm_version: str = "evidence-projection-v1"
    preserve_peak_ratio: float = 1.0
    fabrication_dilation: int = 8
    repair_padding: int = 8
    tile_width: int = 8
    blend_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    score_atol: float = 1e-6
    strict_determinism: bool = True
    strict_union: bool = True
    required_witness_ids: tuple[str, ...] = ()
    concern_threshold: float = 0.62
    concern_min_beta: float = 0.0
    max_auto_changed_fraction: float = 0.35
    max_retained_energy_fraction: float = 1.0
    max_score_evaluations: int = 4096
    max_expansion_rounds: int = 64
    fallback: Literal["observation"] = "observation"

    def __post_init__(self) -> None:
        if not self.algorithm_version:
            raise ValueError("algorithm_version cannot be empty.")
        if not 0.0 <= self.preserve_peak_ratio <= 1.0:
            raise ValueError("preserve_peak_ratio must lie in [0, 1].")
        if min(self.fabrication_dilation, self.repair_padding) < 0:
            raise ValueError("Dilation and padding must be non-negative.")
        if self.tile_width <= 0:
            raise ValueError("tile_width must be positive.")
        grid = tuple(float(value) for value in self.blend_grid)
        if not grid or tuple(sorted(set(grid))) != grid:
            raise ValueError("blend_grid must be strictly increasing and unique.")
        if grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("blend_grid must include 0.0 and 1.0 endpoints.")
        if not all(0.0 <= value <= 1.0 and math.isfinite(value) for value in grid):
            raise ValueError("blend_grid values must be finite values in [0, 1].")
        if self.score_atol < 0 or not math.isfinite(self.score_atol):
            raise ValueError("score_atol must be finite and non-negative.")
        if not 0.0 <= self.concern_threshold <= 1.0:
            raise ValueError("concern_threshold must lie in [0, 1].")
        if self.concern_min_beta not in grid:
            raise ValueError("concern_min_beta must be a member of blend_grid.")
        if not 0.0 <= self.max_auto_changed_fraction <= 1.0:
            raise ValueError("max_auto_changed_fraction must lie in [0, 1].")
        if not 0.0 <= self.max_retained_energy_fraction <= 1.0:
            raise ValueError("max_retained_energy_fraction must lie in [0, 1].")
        if self.max_score_evaluations < 1 or self.max_expansion_rounds < 1:
            raise ValueError("Evaluation and expansion budgets must be positive.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["blend_grid"] = list(self.blend_grid)
        result["required_witness_ids"] = list(self.required_witness_ids)
        return result

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ControllerConfig":
        parsed = dict(values)
        if "blend_grid" in parsed:
            parsed["blend_grid"] = tuple(float(v) for v in parsed["blend_grid"])
        if "required_witness_ids" in parsed:
            parsed["required_witness_ids"] = tuple(parsed["required_witness_ids"])
        return cls(**parsed)


@dataclass(frozen=True)
class WitnessEvent:
    witness_id: str
    event_index: int
    start: int
    end: int
    observation_peak_index: int
    observation_peak_score: float
    required_floor: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WitnessConstraints:
    witness: WitnessSpec
    observation_scores: NDArray[np.float32]
    events: tuple[WitnessEvent, ...]
    fabrication_allowed: NDArray[np.bool_]


@dataclass(frozen=True)
class WitnessVerification:
    witness_id: str
    preservation_passed: bool
    fabrication_passed: bool
    event_output_peaks: tuple[float, ...]
    event_deficits: tuple[float, ...]
    failed_event_indices: tuple[int, ...]
    fabricated_intervals: tuple[tuple[int, int], ...]
    output_scores: NDArray[np.float32]

    @property
    def passed(self) -> bool:
        return self.preservation_passed and self.fabrication_passed

    def to_dict(self, *, include_scores: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "witness_id": self.witness_id,
            "preservation_passed": self.preservation_passed,
            "fabrication_passed": self.fabrication_passed,
            "event_output_peaks": list(self.event_output_peaks),
            "event_deficits": list(self.event_deficits),
            "failed_event_indices": list(self.failed_event_indices),
            "fabricated_intervals": [list(interval) for interval in self.fabricated_intervals],
        }
        if include_scores:
            result["output_scores"] = self.output_scores.tolist()
        return result


@dataclass(frozen=True)
class VerificationResult:
    status: CertificateStatus
    passed: bool
    witnesses: tuple[WitnessVerification, ...]
    limitations: tuple[str, ...]
    error: str | None = None

    def to_dict(self, *, include_scores: bool = False) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "witnesses": [w.to_dict(include_scores=include_scores) for w in self.witnesses],
            "limitations": list(self.limitations),
            "error": self.error,
        }


@dataclass(frozen=True)
class ProjectionInterval:
    start: int
    end: int
    beta: float
    action: Literal["blend", "protect"]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class InterventionCost:
    retained_sample_fraction: float
    retained_observation_energy_fraction: float
    mean_observation_weight: float
    max_observation_weight: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AuditRecord:
    events: tuple[Mapping[str, Any], ...]
    decision_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {"events": [dict(event) for event in self.events], "decision_hash": self.decision_hash}


@dataclass(frozen=True)
class ProjectionResult:
    decision: Decision
    automatic_signal: NDArray[np.float32]
    beta: NDArray[np.float32]
    intervals: tuple[ProjectionInterval, ...]
    certificate: VerificationResult
    auto_committed: bool
    fallback_reason: str | None
    intervention_cost: InterventionCost
    generator_latency_ms: float | None
    controller_latency_ms: float
    total_latency_ms: float | None
    audit: AuditRecord
    provenance: Mapping[str, Any]

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "decision": self.decision,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "certificate": self.certificate.to_dict(include_scores=False),
            "auto_committed": self.auto_committed,
            "fallback_reason": self.fallback_reason,
            "intervention_cost": self.intervention_cost.to_dict(),
            "generator_latency_ms": self.generator_latency_ms,
            "controller_latency_ms": self.controller_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "audit": self.audit.to_dict(),
            "provenance": dict(self.provenance),
        }
        if include_arrays:
            result["automatic_signal"] = self.automatic_signal.tolist()
            result["beta"] = self.beta.tolist()
        return result


class _AuditBuilder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.previous = "0" * 64

    def add(self, kind: str, payload: Mapping[str, Any]) -> None:
        body = {"index": len(self.events), "kind": kind, "payload": dict(payload), "previous": self.previous}
        digest = _sha256_bytes(_canonical_json(body).encode("utf-8"))
        event = dict(body)
        event["hash"] = digest
        self.events.append(event)
        self.previous = digest

    def finish(self, decision_payload: Mapping[str, Any]) -> AuditRecord:
        body = {"last_event_hash": self.previous, "decision": dict(decision_payload)}
        digest = _sha256_bytes(_canonical_json(body).encode("utf-8"))
        return AuditRecord(tuple(self.events), digest)


class EvidencePreservationController:
    """Finite-grid projection with exact-observation abstention fallback."""

    LIMITATIONS = (
        "Certificate applies only to configured deterministic witnesses and frozen thresholds.",
        "Observation-supported evidence may itself be measurement corruption.",
        "Certificate does not establish physical anomaly truth or unseen-detector transfer.",
        "Protecting or blending observation samples can reintroduce measurement corruption.",
    )

    def __init__(self, witnesses: Sequence[WitnessSpec], config: ControllerConfig) -> None:
        if not witnesses:
            raise ValueError("At least one witness is required.")
        ids = tuple(w.witness_id for w in witnesses)
        if len(set(ids)) != len(ids):
            raise ValueError("Witness identifiers must be unique.")
        if config.strict_union and config.required_witness_ids:
            if set(ids) != set(config.required_witness_ids):
                raise ValueError(
                    "Strict-union witness set does not match required_witness_ids: "
                    f"configured={sorted(ids)}, required={sorted(config.required_witness_ids)}"
                )
        self.witnesses = tuple(sorted(witnesses, key=lambda item: item.witness_id))
        self.config = config

    @property
    def provenance(self) -> dict[str, Any]:
        witnesses = [w.provenance_dict() for w in self.witnesses]
        return {
            "algorithm_version": self.config.algorithm_version,
            "config_sha256": self.config.sha256,
            "witnesses": witnesses,
            "witness_set_sha256": _sha256_bytes(_canonical_json(witnesses).encode("utf-8")),
        }

    def _score(self, witness: WitnessSpec, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        scores = np.asarray(witness.scorer(signal.copy()), dtype=np.float32)
        if scores.shape != signal.shape:
            raise ValueError(
                f"Witness {witness.witness_id!r} returned shape {scores.shape}; expected {signal.shape}."
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(f"Witness {witness.witness_id!r} returned non-finite scores.")
        return np.ascontiguousarray(scores)

    def build_constraints(self, observation: ArrayLike) -> tuple[WitnessConstraints, ...]:
        observed = _signal(observation, "observation")
        constraints: list[WitnessConstraints] = []
        for witness in self.witnesses:
            scores = self._score(witness, observed)
            if self.config.strict_determinism:
                repeated = self._score(witness, observed)
                if not np.array_equal(scores, repeated):
                    raise RuntimeError(f"Witness {witness.witness_id!r} is not byte-deterministic.")
            valid = np.ones(observed.size, dtype=bool)
            valid[: witness.context_left] = False
            if witness.context_right:
                valid[observed.size - witness.context_right :] = False
            components = _runs((scores >= witness.threshold) & valid)
            events: list[WitnessEvent] = []
            for index, (start, end) in enumerate(components):
                local = scores[start:end]
                peak_offset = int(np.argmax(local))
                peak_score = float(local[peak_offset])
                events.append(
                    WitnessEvent(
                        witness_id=witness.witness_id,
                        event_index=index,
                        start=start,
                        end=end,
                        observation_peak_index=start + peak_offset,
                        observation_peak_score=peak_score,
                        required_floor=max(
                            float(witness.threshold),
                            float(self.config.preserve_peak_ratio * peak_score),
                        ),
                    )
                )
            allowed = _dilate(components, self.config.fabrication_dilation, observed.size)
            allowed[~valid] = True  # Uncertifiable scorer prefix/suffix is outside the claim.
            constraints.append(WitnessConstraints(witness, scores, tuple(events), allowed))
        return tuple(constraints)

    def _verify_with_constraints(
        self,
        signal: NDArray[np.float32],
        constraints: Sequence[WitnessConstraints],
    ) -> VerificationResult:
        reports: list[WitnessVerification] = []
        try:
            for item in constraints:
                output_scores = self._score(item.witness, signal)
                if self.config.strict_determinism:
                    repeated = self._score(item.witness, signal)
                    if not np.array_equal(output_scores, repeated):
                        raise RuntimeError(
                            f"Witness {item.witness.witness_id!r} is not byte-deterministic."
                        )
                peaks: list[float] = []
                deficits: list[float] = []
                failed: list[int] = []
                for event in item.events:
                    peak = float(np.max(output_scores[event.start : event.end]))
                    deficit = max(0.0, event.required_floor - peak - self.config.score_atol)
                    peaks.append(peak)
                    deficits.append(deficit)
                    if deficit > 0.0:
                        failed.append(event.event_index)
                # Threshold crossing itself is the frozen event decision.  The
                # timing dilation belongs only to matching/non-emergence; no
                # tolerance is allowed to turn a new crossing into a pass.
                fabricated_mask = (
                    (output_scores >= item.witness.threshold)
                    & ~item.fabrication_allowed
                )
                fabricated = _runs(fabricated_mask)
                reports.append(
                    WitnessVerification(
                        witness_id=item.witness.witness_id,
                        preservation_passed=not failed,
                        fabrication_passed=not fabricated,
                        event_output_peaks=tuple(peaks),
                        event_deficits=tuple(deficits),
                        failed_event_indices=tuple(failed),
                        fabricated_intervals=fabricated,
                        output_scores=output_scores,
                    )
                )
        except Exception as error:  # Scoring failure must never approve automatic output.
            return VerificationResult(
                status="unverified",
                passed=False,
                witnesses=tuple(reports),
                limitations=self.LIMITATIONS,
                error=f"{type(error).__name__}: {error}",
            )
        passed = all(report.passed for report in reports)
        return VerificationResult(
            status="passed" if passed else "failed",
            passed=passed,
            witnesses=tuple(reports),
            limitations=self.LIMITATIONS,
        )

    def verify(self, observation: ArrayLike, signal: ArrayLike) -> VerificationResult:
        observed = _signal(observation, "observation")
        output = _signal(signal, "signal")
        if observed.size != output.size:
            raise ValueError("observation and signal lengths must match.")
        try:
            constraints = self.build_constraints(observed)
        except Exception as error:
            return VerificationResult(
                status="unverified",
                passed=False,
                witnesses=(),
                limitations=self.LIMITATIONS,
                error=f"{type(error).__name__}: {error}",
            )
        return self._verify_with_constraints(output, constraints)

    def _violation_intervals(
        self,
        verification: VerificationResult,
        constraints: Sequence[WitnessConstraints],
        size: int,
    ) -> tuple[tuple[int, int, str], ...]:
        by_id = {item.witness.witness_id: item for item in constraints}
        intervals: list[tuple[int, int, str]] = []
        for report in verification.witnesses:
            item = by_id[report.witness_id]
            witness = item.witness
            for event_index in report.failed_event_indices:
                event = item.events[event_index]
                start = event.start - witness.context_left - self.config.repair_padding
                end = event.end + witness.context_right + self.config.repair_padding
                intervals.append((max(0, start), min(size, end), f"{report.witness_id}:retention"))
            for start, end in report.fabricated_intervals:
                intervals.append(
                    (
                        max(0, start - witness.context_left - self.config.repair_padding),
                        min(size, end + witness.context_right + self.config.repair_padding),
                        f"{report.witness_id}:fabrication",
                    )
                )
        return tuple(intervals)

    def _tile_reasons(
        self,
        intervals: Sequence[tuple[int, int, str]],
        size: int,
    ) -> dict[tuple[int, int], set[str]]:
        tiles: dict[tuple[int, int], set[str]] = {}
        width = self.config.tile_width
        for start, end, reason in intervals:
            first = (start // width) * width
            last = ((max(start + 1, end) - 1) // width) * width
            for tile_start in range(first, last + 1, width):
                tile = (tile_start, min(size, tile_start + width))
                tiles.setdefault(tile, set()).add(reason)
        return tiles

    @staticmethod
    def _blend(
        observation: NDArray[np.float32],
        candidate: NDArray[np.float32],
        beta: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        return np.asarray(candidate + beta * (observation - candidate), dtype=np.float32)

    @staticmethod
    def _cost(
        observation: NDArray[np.float32],
        candidate: NDArray[np.float32],
        output: NDArray[np.float32],
        beta: NDArray[np.float32],
    ) -> InterventionCost:
        denominator = float(np.sum(np.square(observation.astype(np.float64) - candidate)))
        numerator = float(np.sum(np.square(output.astype(np.float64) - candidate)))
        energy = 0.0 if denominator <= np.finfo(float).eps else numerator / denominator
        return InterventionCost(
            retained_sample_fraction=float(np.mean(beta > 0.0)),
            retained_observation_energy_fraction=float(np.clip(energy, 0.0, 1.0)),
            mean_observation_weight=float(np.mean(beta)),
            max_observation_weight=float(np.max(beta)),
        )

    def _intervals_from_beta(
        self,
        beta: NDArray[np.float32],
        reasons: Mapping[tuple[int, int], set[str]],
    ) -> tuple[ProjectionInterval, ...]:
        result: list[ProjectionInterval] = []
        for start, end in _runs(beta > 0):
            cursor = start
            while cursor < end:
                value = float(beta[cursor])
                next_cursor = cursor + 1
                while next_cursor < end and float(beta[next_cursor]) == value:
                    next_cursor += 1
                matching: set[str] = set()
                for tile, tile_reasons in reasons.items():
                    if tile[0] < next_cursor and tile[1] > cursor:
                        matching.update(tile_reasons)
                result.append(
                    ProjectionInterval(
                        start=cursor,
                        end=next_cursor,
                        beta=value,
                        action="protect" if value == 1.0 else "blend",
                        reasons=tuple(sorted(matching)),
                    )
                )
                cursor = next_cursor
        return tuple(result)

    def project(
        self,
        observation: ArrayLike,
        base_candidate: ArrayLike,
        *,
        concern: ArrayLike | None = None,
        base_provenance: Mapping[str, Any] | None = None,
        generator_latency_ms: float | None = None,
    ) -> ProjectionResult:
        """Project ``base_candidate`` or abstain to the exact observation.

        Labels are intentionally absent from this API.  ``concern`` can localize
        pre-emptive blending, but it never changes the witness certificate.
        """

        started = time.perf_counter()
        observed = _signal(observation, "observation")
        candidate = _signal(base_candidate, "base_candidate")
        if observed.size != candidate.size:
            raise ValueError("observation and base_candidate lengths must match.")
        concern_values: NDArray[np.float32] | None = None
        if concern is not None:
            concern_values = _signal(concern, "concern")
            if concern_values.size != observed.size:
                raise ValueError("concern length must match the signals.")
            if np.min(concern_values) < 0.0 or np.max(concern_values) > 1.0:
                raise ValueError("concern values must lie in [0, 1].")
        audit = _AuditBuilder()
        provenance = dict(self.provenance)
        provenance["base"] = dict(base_provenance or {})
        provenance["observation_sha256"] = array_sha256(observed)
        provenance["base_candidate_sha256"] = array_sha256(candidate)
        audit.add("inputs", provenance)

        try:
            constraints = self.build_constraints(observed)
        except Exception as error:
            return self._abstain(
                observed,
                candidate,
                reason=f"constraint_build_failed:{type(error).__name__}:{error}",
                audit=audit,
                provenance=provenance,
                constraints=None,
                generator_latency_ms=generator_latency_ms,
                started=started,
            )

        candidate_verification = self._verify_with_constraints(candidate, constraints)
        audit.add(
            "candidate_verification",
            candidate_verification.to_dict(include_scores=False),
        )
        if candidate_verification.status == "unverified":
            return self._abstain(
                observed,
                candidate,
                reason=f"candidate_unverified:{candidate_verification.error}",
                audit=audit,
                provenance=provenance,
                constraints=constraints,
                generator_latency_ms=generator_latency_ms,
                started=started,
            )

        violation_intervals = list(
            self._violation_intervals(candidate_verification, constraints, observed.size)
        )
        if concern_values is not None and self.config.concern_min_beta > 0.0:
            for start, end in _runs(concern_values >= self.config.concern_threshold):
                violation_intervals.append(
                    (
                        max(0, start - self.config.repair_padding),
                        min(observed.size, end + self.config.repair_padding),
                        "concern:preemptive",
                    )
                )

        tile_reasons = self._tile_reasons(violation_intervals, observed.size)
        if candidate_verification.passed and not tile_reasons:
            beta = np.zeros(observed.size, dtype=np.float32)
            return self._finish(
                "accept", candidate, beta, tile_reasons, candidate_verification,
                True, None, observed, candidate, audit, provenance,
                generator_latency_ms, started,
            )

        # This is the predeclared whole-window fallback policy.  Searching for
        # a local projection cannot produce an auto-committed result under a
        # zero intervention budget, so skip the search without changing the
        # returned signal or decision semantics.
        if tile_reasons and (
            self.config.max_auto_changed_fraction == 0.0
            or self.config.max_retained_energy_fraction == 0.0
        ):
            return self._abstain(
                observed,
                candidate,
                reason="zero_intervention_policy",
                audit=audit,
                provenance=provenance,
                constraints=constraints,
                generator_latency_ms=generator_latency_ms,
                started=started,
            )

        beta = np.zeros(observed.size, dtype=np.float32)
        concern_tiles: set[tuple[int, int]] = set()
        for tile, reasons in tile_reasons.items():
            if "concern:preemptive" in reasons:
                concern_tiles.add(tile)
            if reasons == {"concern:preemptive"}:
                beta[tile[0] : tile[1]] = self.config.concern_min_beta
            else:
                beta[tile[0] : tile[1]] = 1.0

        evaluations = 0
        output = self._blend(observed, candidate, beta)
        verification = self._verify_with_constraints(output, constraints)
        evaluations += 1
        rounds = 0
        while not verification.passed and rounds < self.config.max_expansion_rounds:
            rounds += 1
            new_intervals = self._violation_intervals(verification, constraints, observed.size)
            new_tiles = self._tile_reasons(new_intervals, observed.size)
            changed = False
            for tile, reasons in new_tiles.items():
                tile_reasons.setdefault(tile, set()).update(reasons)
                if np.any(beta[tile[0] : tile[1]] < 1.0):
                    beta[tile[0] : tile[1]] = 1.0
                    changed = True
            audit.add(
                "support_expansion",
                {"round": rounds, "new_tiles": [list(tile) for tile in sorted(new_tiles)], "changed": changed},
            )
            if not changed or evaluations >= self.config.max_score_evaluations:
                break
            output = self._blend(observed, candidate, beta)
            verification = self._verify_with_constraints(output, constraints)
            evaluations += 1

        if not verification.passed:
            return self._abstain(
                observed,
                candidate,
                reason=("score_budget_exhausted" if evaluations >= self.config.max_score_evaluations else "local_projection_infeasible"),
                audit=audit,
                provenance=provenance,
                constraints=constraints,
                generator_latency_ms=generator_latency_ms,
                started=started,
            )

        # Deterministic coordinate release.  The certificate is recomputed over
        # the complete window for every trial; no monotonic scorer assumption is made.
        grid = self.config.blend_grid
        active_tiles = sorted(tile_reasons)
        changed = True
        while changed and evaluations < self.config.max_score_evaluations:
            changed = False
            for tile in active_tiles:
                current = float(beta[tile[0]])
                lower_bound = self.config.concern_min_beta if tile in concern_tiles else 0.0
                choices = [value for value in grid if lower_bound <= value < current]
                accepted: tuple[float, VerificationResult] | None = None
                original = beta[tile[0] : tile[1]].copy()
                for value in choices:
                    beta[tile[0] : tile[1]] = value
                    trial = self._blend(observed, candidate, beta)
                    trial_verification = self._verify_with_constraints(trial, constraints)
                    evaluations += 1
                    if trial_verification.passed:
                        accepted = (value, trial_verification)
                        output = trial
                        break
                    if evaluations >= self.config.max_score_evaluations:
                        break
                if accepted is None:
                    beta[tile[0] : tile[1]] = original
                else:
                    verification = accepted[1]
                    changed = True
                    audit.add(
                        "tile_release",
                        {"tile": list(tile), "from_beta": current, "to_beta": accepted[0]},
                    )
                if evaluations >= self.config.max_score_evaluations:
                    break

        output = self._blend(observed, candidate, beta)
        verification = self._verify_with_constraints(output, constraints)
        evaluations += 1
        cost = self._cost(observed, candidate, output, beta)
        audit.add(
            "final_verification",
            {
                "score_evaluations": evaluations,
                "verification": verification.to_dict(include_scores=False),
                "cost": cost.to_dict(),
            },
        )
        if not verification.passed:
            return self._abstain(
                observed, candidate, reason="final_postcondition_failed", audit=audit,
                provenance=provenance, constraints=constraints,
                generator_latency_ms=generator_latency_ms, started=started,
            )
        if cost.retained_sample_fraction > self.config.max_auto_changed_fraction + 1e-12:
            return self._abstain(
                observed, candidate, reason="changed_fraction_budget_exceeded", audit=audit,
                provenance=provenance, constraints=constraints,
                generator_latency_ms=generator_latency_ms, started=started,
            )
        if cost.retained_observation_energy_fraction > self.config.max_retained_energy_fraction + 1e-12:
            return self._abstain(
                observed, candidate, reason="retained_energy_budget_exceeded", audit=audit,
                provenance=provenance, constraints=constraints,
                generator_latency_ms=generator_latency_ms, started=started,
            )

        values = set(float(value) for value in beta[beta > 0])
        if values == {1.0}:
            decision: Decision = "protect"
        elif 1.0 in values:
            decision = "mixed"
        else:
            decision = "blend"
        return self._finish(
            decision, output, beta, tile_reasons, verification, True, None,
            observed, candidate, audit, provenance, generator_latency_ms, started,
        )

    def _abstain(
        self,
        observation: NDArray[np.float32],
        candidate: NDArray[np.float32],
        *,
        reason: str,
        audit: _AuditBuilder,
        provenance: Mapping[str, Any],
        constraints: Sequence[WitnessConstraints] | None,
        generator_latency_ms: float | None,
        started: float,
    ) -> ProjectionResult:
        output = observation.copy()  # Required strict fallback: byte-exact observation.
        beta = np.ones(observation.size, dtype=np.float32)
        if constraints is None:
            certificate = VerificationResult(
                status="unverified", passed=False, witnesses=(),
                limitations=self.LIMITATIONS, error=reason,
            )
        else:
            certificate = self._verify_with_constraints(output, constraints)
        audit.add(
            "abstain",
            {
                "reason": reason,
                "exact_observation_fallback": bool(np.array_equal(output, observation)),
                "certificate": certificate.to_dict(include_scores=False),
            },
        )
        all_reasons = {(0, observation.size): {"abstain:exact_observation_fallback"}}
        return self._finish(
            "abstain", output, beta, all_reasons, certificate, False, reason,
            observation, candidate, audit, provenance, generator_latency_ms, started,
        )

    def _finish(
        self,
        decision: Decision,
        output: NDArray[np.float32],
        beta: NDArray[np.float32],
        reasons: Mapping[tuple[int, int], set[str]],
        certificate: VerificationResult,
        auto_committed: bool,
        fallback_reason: str | None,
        observation: NDArray[np.float32],
        candidate: NDArray[np.float32],
        audit: _AuditBuilder,
        provenance: Mapping[str, Any],
        generator_latency_ms: float | None,
        started: float,
    ) -> ProjectionResult:
        output = np.ascontiguousarray(output, dtype=np.float32)
        beta = np.ascontiguousarray(beta, dtype=np.float32)
        cost = self._cost(observation, candidate, output, beta)
        intervals = self._intervals_from_beta(beta, reasons)
        decision_payload = {
            "decision": decision,
            "auto_committed": auto_committed,
            "fallback_reason": fallback_reason,
            "output_sha256": array_sha256(output),
            "beta_sha256": array_sha256(beta),
            "certificate_status": certificate.status,
            "certificate_passed": certificate.passed,
            "cost": cost.to_dict(),
            "intervals": [interval.to_dict() for interval in intervals],
        }
        final_audit = audit.finish(decision_payload)
        controller_latency = (time.perf_counter() - started) * 1000.0
        total_latency = None if generator_latency_ms is None else generator_latency_ms + controller_latency
        return ProjectionResult(
            decision=decision,
            automatic_signal=output,
            beta=beta,
            intervals=intervals,
            certificate=certificate,
            auto_committed=auto_committed,
            fallback_reason=fallback_reason,
            intervention_cost=cost,
            generator_latency_ms=generator_latency_ms,
            controller_latency_ms=controller_latency,
            total_latency_ms=total_latency,
            audit=final_audit,
            provenance=dict(provenance),
        )

    def new_session(
        self,
        observation: ArrayLike,
        base_candidate: ArrayLike,
        automatic: ProjectionResult,
    ) -> "ControlledSignalSession":
        return ControlledSignalSession(self, observation, base_candidate, automatic)


@dataclass(frozen=True)
class OverrideRecord:
    revision: int
    action: str
    start: int
    end: int
    beta: float | None
    previous_signal_sha256: str
    resulting_signal_sha256: str
    certificate_status: CertificateStatus
    previous_record_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ControlledSignalSession:
    """Human override session with an immutable automatic baseline."""

    def __init__(
        self,
        controller: EvidencePreservationController,
        observation: ArrayLike,
        base_candidate: ArrayLike,
        automatic: ProjectionResult,
    ) -> None:
        self.controller = controller
        self.observation = _signal(observation, "observation")
        self.base_candidate = _signal(base_candidate, "base_candidate")
        self.automatic_baseline = _signal(automatic.automatic_signal, "automatic_signal")
        if not (
            self.observation.size == self.base_candidate.size == self.automatic_baseline.size
        ):
            raise ValueError("All session signals must have equal length.")
        self.current = self.automatic_baseline.copy()
        self._states: list[NDArray[np.float32]] = []
        self._records: list[OverrideRecord] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def records(self) -> tuple[OverrideRecord, ...]:
        return tuple(self._records)

    def _check_revision(self, expected_revision: int | None) -> None:
        if expected_revision is not None and int(expected_revision) != self._revision:
            raise RuntimeError(
                f"Stale intervention revision: expected {expected_revision}, current {self._revision}."
            )

    def _record(
        self,
        action: str,
        start: int,
        end: int,
        beta: float | None,
        previous: NDArray[np.float32],
    ) -> VerificationResult:
        verification = self.controller.verify(self.observation, self.current)
        status: CertificateStatus = "overridden" if not verification.passed else "passed"
        prior_hash = self._records[-1].record_hash if self._records else "0" * 64
        body = {
            "revision": self._revision,
            "action": action,
            "start": start,
            "end": end,
            "beta": beta,
            "previous_signal_sha256": array_sha256(previous),
            "resulting_signal_sha256": array_sha256(self.current),
            "certificate_status": status,
            "previous_record_hash": prior_hash,
        }
        record_hash = _sha256_bytes(_canonical_json(body).encode("utf-8"))
        self._records.append(OverrideRecord(**body, record_hash=record_hash))
        if status == "overridden":
            return VerificationResult(
                status="overridden", passed=False, witnesses=verification.witnesses,
                limitations=verification.limitations, error=verification.error,
            )
        return verification

    def apply(
        self,
        action: Literal["accept_candidate", "protect", "blend", "restore_automatic"],
        start: int | None = None,
        end: int | None = None,
        *,
        beta: float = 0.5,
        expected_revision: int | None = None,
    ) -> tuple[NDArray[np.float32], VerificationResult]:
        self._check_revision(expected_revision)
        start_index = 0 if start is None else int(start)
        end_index = self.current.size if end is None else int(end)
        if not 0 <= start_index < end_index <= self.current.size:
            raise ValueError("Intervention interval must satisfy 0 <= start < end <= length.")
        previous = self.current.copy()
        self._states.append(previous)
        if action == "restore_automatic":
            self.current = self.automatic_baseline.copy()
            start_index, end_index = 0, self.current.size
            beta_value: float | None = None
        elif action == "accept_candidate":
            self.current[start_index:end_index] = self.base_candidate[start_index:end_index]
            beta_value = None
        elif action == "protect":
            self.current[start_index:end_index] = self.observation[start_index:end_index]
            beta_value = None
        elif action == "blend":
            if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
                raise ValueError("beta must be a finite value in [0, 1].")
            self.current[start_index:end_index] = (
                beta * self.observation[start_index:end_index]
                + (1.0 - beta) * self.base_candidate[start_index:end_index]
            )
            beta_value = float(beta)
        else:
            self._states.pop()
            raise ValueError(f"Unsupported intervention action: {action!r}")
        self.current = np.ascontiguousarray(self.current, dtype=np.float32)
        self._revision += 1
        verification = self._record(action, start_index, end_index, beta_value, previous)
        return self.current.copy(), verification

    def revert(
        self, *, expected_revision: int | None = None
    ) -> tuple[NDArray[np.float32], VerificationResult]:
        self._check_revision(expected_revision)
        if not self._states:
            raise RuntimeError("No human intervention is available to revert.")
        previous = self.current.copy()
        self.current = self._states.pop()
        self._revision += 1
        verification = self._record("revert", 0, self.current.size, None, previous)
        return self.current.copy(), verification


def frozen_config_payload(config: ControllerConfig, *, development_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a hash-addressed config artifact for an external confirmation runner."""

    body = {
        "schema_version": 1,
        "controller_config": config.to_dict(),
        "controller_config_sha256": config.sha256,
        "development_metadata": dict(development_metadata),
        "confirmation_results": None,
    }
    body["artifact_sha256"] = _sha256_bytes(_canonical_json(body).encode("utf-8"))
    return body


def load_frozen_config(payload: Mapping[str, Any]) -> ControllerConfig:
    """Validate and load a frozen controller configuration artifact."""

    values = dict(payload)
    supplied_artifact_hash = values.pop("artifact_sha256", None)
    expected_artifact_hash = _sha256_bytes(_canonical_json(values).encode("utf-8"))
    if supplied_artifact_hash != expected_artifact_hash:
        raise ValueError("Frozen controller artifact SHA-256 mismatch.")
    config = ControllerConfig.from_dict(values["controller_config"])
    if values.get("controller_config_sha256") != config.sha256:
        raise ValueError("Frozen controller configuration SHA-256 mismatch.")
    return config
