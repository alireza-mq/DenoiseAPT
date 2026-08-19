"""Evidence-gated classical/DenoiseAPT hybrid routing.

This module is intentionally separate from the frozen evidence controller.  It
uses that controller as an immutable witness interface, but it does not alter
the controller or any historical protocol artifact.

The routing statement is deliberately narrow: a classical candidate is used
where it satisfies the configured A/B witness contract.  Failed preservation
or output-only-evidence regions are replaced by a DenoiseAPT repair source,
with a short raised-cosine halo to avoid hard seams.  This is not a claim that
the remaining samples contain no physical anomaly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .evidence_controller import (
    EvidencePreservationController,
    VerificationResult,
    WitnessConstraints,
)


HYBRID_ALGORITHM_VERSION = "evidence-gated-classical-dapt-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _signal(value: ArrayLike, name: str) -> NDArray[np.float32]:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 1 or result.size < 32 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain at least 32 finite samples.")
    return np.ascontiguousarray(result)


def reflect_moving_average(values: ArrayLike, width: int = 9) -> NDArray[np.float32]:
    """Return the live UI's odd-width, reflect-padded moving average."""

    signal = _signal(values, "values")
    width = max(3, int(width) | 1)
    pad = width // 2
    padded = np.pad(signal.astype(np.float64), pad, mode="reflect")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.ascontiguousarray(
        np.convolve(padded, kernel, mode="valid"), dtype=np.float32
    )


@dataclass(frozen=True)
class HybridConfig:
    """Frozen routing choices for the first hybrid implementation."""

    algorithm_version: str = HYBRID_ALGORITHM_VERSION
    classical_filter: str = "moving_average_w9_reflect"
    classical_width: int = 9
    halo_width: int = 8
    halo_weights: tuple[float, ...] = (
        0.96984631,
        0.88302222,
        0.75,
        0.58682412,
        0.41317591,
        0.25,
        0.11697778,
        0.03015369,
    )
    required_witness_ids: tuple[str, ...] = (
        "A_causal_mlp",
        "B_causal_conv",
    )
    final_whole_window_projection: bool = True
    review_only_outside_threshold_scope: bool = True

    def __post_init__(self) -> None:
        if self.algorithm_version != HYBRID_ALGORITHM_VERSION:
            raise ValueError("Unsupported hybrid algorithm version.")
        if self.classical_filter != "moving_average_w9_reflect":
            raise ValueError("The v1 hybrid requires the exact live MA9 filter.")
        if self.classical_width != 9:
            raise ValueError("The v1 hybrid classical width is frozen at 9.")
        if self.halo_width < 0:
            raise ValueError("halo_width must be non-negative.")
        if self.halo_width != len(self.halo_weights):
            raise ValueError("halo_width must match the frozen halo vector.")
        weights = np.asarray(self.halo_weights, dtype=np.float32)
        if (
            weights.ndim != 1
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or np.any(weights >= 1.0)
            or np.any(np.diff(weights) >= 0.0)
        ):
            raise ValueError("halo_weights must be finite and strictly decreasing in (0, 1).")
        if tuple(self.required_witness_ids) != (
            "A_causal_mlp",
            "B_causal_conv",
        ):
            raise ValueError("The v1 hybrid requires the strict A/B witness set.")
        if not self.final_whole_window_projection:
            raise ValueError("The v1 hybrid requires a final whole-window projection.")
        if not self.review_only_outside_threshold_scope:
            raise ValueError("Out-of-scope requests must remain review-only.")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["required_witness_ids"] = list(self.required_witness_ids)
        result["halo_weights"] = list(self.halo_weights)
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HybridConfig":
        parsed = dict(value)
        if "required_witness_ids" in parsed:
            parsed["required_witness_ids"] = tuple(parsed["required_witness_ids"])
        if "halo_weights" in parsed:
            parsed["halo_weights"] = tuple(float(value) for value in parsed["halo_weights"])
        return cls(**parsed)


LIVE_HYBRID_CONFIG = HybridConfig()


@dataclass(frozen=True)
class HybridRouteInterval:
    start: int
    end: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class HybridCandidateResult:
    classical: NDArray[np.float32]
    requested_repair_source: NDArray[np.float32]
    effective_repair_source: NDArray[np.float32]
    candidate: NDArray[np.float32]
    routing_weight: NDArray[np.float32]
    hard_intervals: tuple[HybridRouteInterval, ...]
    classical_verification: VerificationResult
    repair_source_verification: VerificationResult | None
    candidate_verification: VerificationResult
    repair_source_kind: str
    config: HybridConfig

    @property
    def hard_routed_fraction(self) -> float:
        return float(np.mean(self.routing_weight == 1.0))

    @property
    def nonzero_routed_fraction(self) -> float:
        return float(np.mean(self.routing_weight > 0.0))

    @property
    def mean_denoiseapt_weight(self) -> float:
        return float(np.mean(self.routing_weight))

    def routing_payload(self) -> dict[str, Any]:
        return {
            "algorithm_version": self.config.algorithm_version,
            "config_sha256": self.config.sha256,
            "classical_filter": self.config.classical_filter,
            "repair_source_kind": self.repair_source_kind,
            "hard_intervals": [item.to_dict() for item in self.hard_intervals],
            "hard_routed_fraction": self.hard_routed_fraction,
            "nonzero_routed_fraction": self.nonzero_routed_fraction,
            "mean_denoiseapt_weight": self.mean_denoiseapt_weight,
            "classical_witness_check": self.classical_verification.to_dict(),
            "repair_source_witness_check": (
                None
                if self.repair_source_verification is None
                else self.repair_source_verification.to_dict()
            ),
            "hybrid_candidate_witness_check": self.candidate_verification.to_dict(),
            "interpretation": (
                "Classical output is retained where it satisfies the configured A/B "
                "witness contract; routed support is not a physical-anomaly label."
            ),
        }


def _merge_reasoned(
    intervals: Sequence[tuple[int, int, str]], size: int, tile_width: int
) -> tuple[HybridRouteInterval, ...]:
    tiles: dict[tuple[int, int], set[str]] = {}
    for start, end, reason in intervals:
        start = max(0, min(int(start), size))
        end = max(start, min(int(end), size))
        if end <= start:
            continue
        first = (start // tile_width) * tile_width
        last = ((end - 1) // tile_width) * tile_width
        for tile_start in range(first, last + 1, tile_width):
            tile = (tile_start, min(size, tile_start + tile_width))
            tiles.setdefault(tile, set()).add(str(reason))
    if not tiles:
        return ()
    result: list[HybridRouteInterval] = []
    for (start, end), reasons in sorted(tiles.items()):
        if result and start == result[-1].end:
            previous = result[-1]
            result[-1] = HybridRouteInterval(
                previous.start,
                end,
                tuple(sorted(set(previous.reasons).union(reasons))),
            )
        else:
            result.append(HybridRouteInterval(start, end, tuple(sorted(reasons))))
    return tuple(result)


def _failure_intervals(
    verification: VerificationResult,
    constraints: Sequence[WitnessConstraints],
    controller: EvidencePreservationController,
    size: int,
) -> tuple[HybridRouteInterval, ...]:
    if verification.status == "unverified":
        return (HybridRouteInterval(0, size, ("witness_check:unverified",)),)
    by_id = {item.witness.witness_id: item for item in constraints}
    raw: list[tuple[int, int, str]] = []
    for report in verification.witnesses:
        item = by_id.get(report.witness_id)
        if item is None:
            return (HybridRouteInterval(0, size, ("witness_set:mismatch",)),)
        witness = item.witness
        for event_index in report.failed_event_indices:
            event = item.events[event_index]
            raw.append(
                (
                    event.start - witness.context_left - controller.config.repair_padding,
                    event.end + witness.context_right + controller.config.repair_padding,
                    f"{report.witness_id}:retention",
                )
            )
        for start, end in report.fabricated_intervals:
            raw.append(
                (
                    start - witness.context_left - controller.config.repair_padding,
                    end + witness.context_right + controller.config.repair_padding,
                    f"{report.witness_id}:output_only",
                )
            )
    if not verification.passed and not raw:
        raw.append((0, size, "witness_check:failed_without_localization"))
    return _merge_reasoned(raw, size, controller.config.tile_width)


def _raised_cosine_weight(
    size: int,
    intervals: Sequence[HybridRouteInterval],
    halo_weights: Sequence[float],
) -> NDArray[np.float32]:
    weight = np.zeros(size, dtype=np.float32)
    for item in intervals:
        weight[item.start : item.end] = 1.0
    if not halo_weights:
        return weight
    for item in intervals:
        for distance, frozen_value in enumerate(halo_weights, start=1):
            value = float(np.float32(frozen_value))
            left = item.start - distance
            right = item.end - 1 + distance
            if left >= 0:
                weight[left] = max(weight[left], value)
            if right < size:
                weight[right] = max(weight[right], value)
    return np.ascontiguousarray(weight)


def build_evidence_gated_candidate(
    observation: ArrayLike,
    classical: ArrayLike,
    denoiseapt_repair_source: ArrayLike,
    controller: EvidencePreservationController,
    *,
    config: HybridConfig = LIVE_HYBRID_CONFIG,
) -> HybridCandidateResult:
    """Build one deterministic hybrid candidate without using labels.

    The returned witness checks are diagnostic unless the caller separately
    establishes that the request is inside the frozen threshold scope.
    """

    observed = _signal(observation, "observation")
    classical_signal = _signal(classical, "classical")
    requested_source = _signal(denoiseapt_repair_source, "denoiseapt_repair_source")
    if not (
        observed.size == classical_signal.size == requested_source.size
    ):
        raise ValueError("observation, classical, and repair source lengths must match.")
    controller_ids = tuple(witness.witness_id for witness in controller.witnesses)
    if tuple(sorted(controller_ids)) != tuple(sorted(config.required_witness_ids)):
        raise ValueError("Hybrid controller does not contain the frozen A/B witness set.")

    classical_verification = controller.verify(observed, classical_signal)
    if classical_verification.status == "unverified":
        # A scorer failure cannot justify either expert. Return the exact
        # observation as the routed candidate and preserve the unverified
        # status for the caller's fail-closed decision.
        weight = np.ones(observed.size, dtype=np.float32)
        return HybridCandidateResult(
            classical=classical_signal,
            requested_repair_source=requested_source,
            effective_repair_source=observed.copy(),
            candidate=observed.copy(),
            routing_weight=weight,
            hard_intervals=(
                HybridRouteInterval(
                    0, observed.size, ("witness_check:unverified_observation_fallback",)
                ),
            ),
            classical_verification=classical_verification,
            repair_source_verification=classical_verification,
            candidate_verification=classical_verification,
            repair_source_kind="observation_fallback_unverified",
            config=config,
        )
    try:
        constraints = controller.build_constraints(observed)
    except Exception as error:
        unverified = VerificationResult(
            status="unverified",
            passed=False,
            witnesses=(),
            limitations=controller.LIMITATIONS,
            error=f"{type(error).__name__}: {error}",
        )
        weight = np.ones(observed.size, dtype=np.float32)
        return HybridCandidateResult(
            classical=classical_signal,
            requested_repair_source=requested_source,
            effective_repair_source=observed.copy(),
            candidate=observed.copy(),
            routing_weight=weight,
            hard_intervals=(
                HybridRouteInterval(
                    0, observed.size, ("constraint_rebuild:observation_fallback",)
                ),
            ),
            classical_verification=unverified,
            repair_source_verification=unverified,
            candidate_verification=unverified,
            repair_source_kind="observation_fallback_unverified",
            config=config,
        )
    intervals = _failure_intervals(
        classical_verification, constraints, controller, observed.size
    )

    repair_verification: VerificationResult | None = None
    repair_source = requested_source
    repair_kind = "denoiseapt"
    if intervals:
        repair_verification = controller.verify(observed, requested_source)
        if not repair_verification.passed:
            repair_source = observed.copy()
            repair_kind = "observation_fallback"
            repair_verification = controller.verify(observed, repair_source)
    else:
        repair_kind = "not_needed"

    weight = _raised_cosine_weight(observed.size, intervals, config.halo_weights)
    candidate = np.asarray(
        classical_signal + weight * (repair_source - classical_signal),
        dtype=np.float32,
    )
    candidate = np.ascontiguousarray(candidate)
    candidate_verification = (
        classical_verification
        if not intervals
        else controller.verify(observed, candidate)
    )
    return HybridCandidateResult(
        classical=classical_signal,
        requested_repair_source=requested_source,
        effective_repair_source=repair_source,
        candidate=candidate,
        routing_weight=weight,
        hard_intervals=intervals,
        classical_verification=classical_verification,
        repair_source_verification=repair_verification,
        candidate_verification=candidate_verification,
        repair_source_kind=repair_kind,
        config=config,
    )
