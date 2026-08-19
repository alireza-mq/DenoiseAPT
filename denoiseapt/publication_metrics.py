"""Publication-only anomaly evaluation with pinned upstream provenance.

This module is deliberately separate from :mod:`denoiseapt.metrics`, whose
``vus_pr_approximation`` is intended only for responsive dashboard feedback.
The VUS-PR routine below is a dependency-light NumPy port of the authoritative
TSB-AD 1.5 implementation.  It preserves the algorithm used by
``TSB_AD.evaluation.basic_metrics.generate_curve`` at the pinned commit named
in :data:`TSB_AD_VUS_PROVENANCE`.

The upstream implementation is Apache-2.0 licensed.  See
``LICENSES/THIRD_PARTY_NOTICES.md``.  No code path in this module silently substitutes
the dashboard approximation for the publication metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class VUSProvenance:
    """Exact upstream identity for the ported VUS-PR implementation."""

    package: str
    package_version: str
    repository: str
    commit: str
    commit_date_utc: str
    source_file: str
    source_sha256: str
    wrapper_file: str
    wrapper_sha256: str
    license: str
    paper_doi: str
    implementation: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


TSB_AD_VUS_PROVENANCE = VUSProvenance(
    package="TSB-AD",
    package_version="1.5",
    repository="https://github.com/TheDatumOrg/TSB-AD",
    commit="e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48",
    commit_date_utc="2026-07-02T21:59:49Z",
    source_file="TSB_AD/evaluation/basic_metrics.py",
    # SHA-256 values are over the canonical Git blob bytes at ``commit``
    # (before any platform-specific checkout newline conversion).
    source_sha256="1fcddedf5ada1d5221f39ee568c7fddb9e7181bd7e1c19636c2cbecf40c97707",
    wrapper_file="TSB_AD/evaluation/metrics.py",
    wrapper_sha256="13957d15d3ebb2b743a4bebfa3a68af31e82a58ecc568557b4081ad90e38094d",
    license="Apache-2.0",
    paper_doi="10.14778/3551793.3551830",
    implementation="minimal NumPy port of the pinned TSB-AD VUS-PR algorithm",
)


@dataclass(frozen=True)
class OfficialVUSPRResult:
    """A VUS-PR value together with every setting needed to reproduce it."""

    value: float
    max_buffer: int
    threshold_count: int
    pr_auc_by_buffer: tuple[float, ...]
    provenance: VUSProvenance = TSB_AD_VUS_PROVENANCE

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": "VUS-PR",
            "value": self.value,
            "max_buffer": self.max_buffer,
            "threshold_count": self.threshold_count,
            "pr_auc_by_buffer": list(self.pr_auc_by_buffer),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True)
class CountRate:
    """A count ratio whose value is ``None`` when no case is eligible."""

    numerator: int
    denominator: int
    rate: float | None

    @classmethod
    def from_counts(cls, numerator: int, denominator: int) -> "CountRate":
        if numerator < 0 or denominator < 0 or numerator > denominator:
            raise ValueError("count ratios require 0 <= numerator <= denominator")
        rate = None if denominator == 0 else float(numerator / denominator)
        return cls(numerator=numerator, denominator=denominator, rate=rate)

    def to_dict(self) -> dict[str, int | float | str | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "rate": self.rate,
            "status": "not_applicable" if self.rate is None else "defined",
        }


@dataclass(frozen=True)
class EventDecisionMetrics:
    """Auditable transitions between reference, observed, and restored events.

    ``reference_retention`` asks whether labelled events detected on the clean
    reference remain detectable after denoising.  ``denoising_erasure`` asks
    whether labelled events detected in the noisy observation disappear after
    denoising.  ``denoising_recovery`` is restricted to labelled events that
    are detectable on the clean reference but missed in the noisy observation.
    ``false_event_generation`` counts restored predicted intervals that overlap
    neither a label nor a predicted interval in the noisy observation.
    """

    labelled_events: int
    reference_detectable_events: int
    observation_detectable_events: int
    candidate_detectable_events: int
    candidate_predicted_events: int
    reference_retention: CountRate
    denoising_erasure: CountRate
    denoising_recovery: CountRate
    false_event_generation: CountRate

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": {
                "labelled_events": self.labelled_events,
                "reference_detectable_events": self.reference_detectable_events,
                "observation_detectable_events": self.observation_detectable_events,
                "candidate_detectable_events": self.candidate_detectable_events,
                "candidate_predicted_events": self.candidate_predicted_events,
            },
            "reference_retention": self.reference_retention.to_dict(),
            "denoising_erasure": self.denoising_erasure.to_dict(),
            "denoising_recovery": self.denoising_recovery.to_dict(),
            "false_event_generation": self.false_event_generation.to_dict(),
        }


def official_vus_pr(
    labels: Sequence[bool] | np.ndarray,
    scores: ArrayLike,
    *,
    max_buffer: int,
    threshold_count: int = 250,
) -> OfficialVUSPRResult:
    """Compute VUS-PR using the pinned TSB-AD 1.5 algorithm.

    Higher ``scores`` must indicate greater anomalousness. ``max_buffer`` is
    the TSB-AD ``slidingWindow`` argument: the implementation averages the
    range-aware PR area over every integer buffer from zero through this value.
    The value must be chosen before inspecting method results and recorded in
    the experiment manifest.  This function intentionally has no approximate
    fallback.
    """

    truth = _binary_array(labels, "labels")
    values = _finite_array(scores, "scores")
    if len(truth) != len(values):
        raise ValueError("labels and scores must have equal lengths")
    if not isinstance(max_buffer, (int, np.integer)) or isinstance(
        max_buffer, bool
    ):
        raise TypeError("max_buffer must be an integer")
    if not isinstance(threshold_count, (int, np.integer)) or isinstance(
        threshold_count, bool
    ):
        raise TypeError("threshold_count must be an integer")
    max_buffer = int(max_buffer)
    threshold_count = int(threshold_count)
    if max_buffer < 0 or max_buffer >= len(truth):
        raise ValueError("max_buffer must satisfy 0 <= max_buffer < series length")
    if threshold_count < 2:
        raise ValueError("threshold_count must be at least two")
    if not np.any(truth):
        raise ValueError("VUS-PR is undefined when labels contain no anomaly")
    if np.all(truth):
        raise ValueError("VUS-PR requires at least one normal point")

    # The following computation is an equivalent, dependency-light port of
    # RangeAUC_volume_opt() and generate_curve() in the pinned source file.
    anomaly_events = _inclusive_events(truth)
    evaluation_extent = _merge_buffered_events(len(truth), anomaly_events, max_buffer)
    positive_points = float(np.sum(truth))

    descending_scores = -np.sort(-values)
    threshold_indices = np.linspace(0, len(values) - 1, threshold_count).astype(int)
    thresholds = descending_scores[threshold_indices]
    predicted_counts = np.asarray(
        [np.sum(values >= threshold) for threshold in thresholds], dtype=np.float64
    )

    pr_auc_by_buffer = np.zeros(max_buffer + 1, dtype=np.float64)
    for buffer in range(max_buffer + 1):
        extended_labels = _extend_labels(truth, anomaly_events, buffer)
        extended_events = _merge_buffered_events(len(truth), anomaly_events, buffer)
        tpr = np.zeros(threshold_count + 2, dtype=np.float64)
        precision = np.ones(threshold_count + 1, dtype=np.float64)

        for index, threshold in enumerate(thresholds):
            prediction = values >= threshold
            weighted_labels = extended_labels.copy()
            existence = 0
            for start, stop in extended_events:
                weighted_labels[start : stop + 1] = (
                    extended_labels[start : stop + 1] * prediction[start : stop + 1]
                )
                if np.any(prediction[start : stop + 1]):
                    existence += 1
            for start, stop in anomaly_events:
                weighted_labels[start : stop + 1] = 1.0

            true_positive = 0.0
            weighted_positive_count = 0.0
            for start, stop in evaluation_extent:
                true_positive += float(
                    np.dot(
                        weighted_labels[start : stop + 1],
                        prediction[start : stop + 1],
                    )
                )
                weighted_positive_count += float(
                    np.sum(weighted_labels[start : stop + 1])
                )

            existence_ratio = existence / len(extended_events)
            adjusted_positives = (positive_points + weighted_positive_count) / 2.0
            recall = min(true_positive / adjusted_positives, 1.0)
            tpr[index + 1] = recall * existence_ratio
            precision[index + 1] = true_positive / predicted_counts[index]

        tpr[-1] = 1.0
        recall_width = tpr[1:-1] - tpr[:-2]
        precision_height = precision[1:]
        pr_auc_by_buffer[buffer] = float(np.dot(recall_width, precision_height))

    value = float(np.sum(pr_auc_by_buffer) / len(pr_auc_by_buffer))
    if not math.isfinite(value):
        raise RuntimeError(
            "the pinned TSB-AD VUS-PR computation produced a non-finite value"
        )
    return OfficialVUSPRResult(
        value=value,
        max_buffer=max_buffer,
        threshold_count=threshold_count,
        pr_auc_by_buffer=tuple(float(item) for item in pr_auc_by_buffer),
    )


def event_decision_metrics(
    labels: Sequence[bool] | np.ndarray,
    reference_decisions: Sequence[bool] | np.ndarray,
    observation_decisions: Sequence[bool] | np.ndarray,
    candidate_decisions: Sequence[bool] | np.ndarray,
) -> EventDecisionMetrics:
    """Count event-level preservation transitions from explicit decisions.

    Inputs must already be binary.  Publication code should create these
    decisions with one threshold selected on held-out validation data and then
    use that same threshold for every compared denoiser.
    """

    truth = _binary_array(labels, "labels")
    reference = _binary_array(reference_decisions, "reference_decisions")
    observation = _binary_array(observation_decisions, "observation_decisions")
    candidate = _binary_array(candidate_decisions, "candidate_decisions")
    if not (len(truth) == len(reference) == len(observation) == len(candidate)):
        raise ValueError("labels and all decision arrays must have equal lengths")

    labelled = _inclusive_events(truth)
    state: list[tuple[bool, bool, bool]] = []
    for start, stop in labelled:
        state.append(
            (
                bool(np.any(reference[start : stop + 1])),
                bool(np.any(observation[start : stop + 1])),
                bool(np.any(candidate[start : stop + 1])),
            )
        )

    reference_detectable = sum(item[0] for item in state)
    observation_detectable = sum(item[1] for item in state)
    candidate_detectable = sum(item[2] for item in state)
    retained = sum(
        reference_hit and candidate_hit
        for reference_hit, _, candidate_hit in state
    )
    erased = sum(
        observation_hit and not candidate_hit
        for _, observation_hit, candidate_hit in state
    )
    recovery_eligible = sum(
        reference_hit and not observation_hit
        for reference_hit, observation_hit, _ in state
    )
    recovered = sum(
        reference_hit and not observation_hit and candidate_hit
        for reference_hit, observation_hit, candidate_hit in state
    )

    candidate_events = _inclusive_events(candidate)
    false_generated = sum(
        not np.any(truth[start : stop + 1])
        and not np.any(observation[start : stop + 1])
        for start, stop in candidate_events
    )
    return EventDecisionMetrics(
        labelled_events=len(labelled),
        reference_detectable_events=reference_detectable,
        observation_detectable_events=observation_detectable,
        candidate_detectable_events=candidate_detectable,
        candidate_predicted_events=len(candidate_events),
        reference_retention=CountRate.from_counts(retained, reference_detectable),
        denoising_erasure=CountRate.from_counts(erased, observation_detectable),
        denoising_recovery=CountRate.from_counts(recovered, recovery_eligible),
        false_event_generation=CountRate.from_counts(false_generated, len(candidate_events)),
    )


def median_anomaly_length(labels: Sequence[bool] | np.ndarray) -> int:
    """Return the median labelled-event length as an explicit VUS buffer policy.

    The upstream VUS project documents median labelled anomaly length as one
    permissible external-knowledge policy.  TSB-AD's benchmark runner instead
    estimates periodicity from the signal.  Whichever policy is used must be
    fixed and recorded consistently for all methods.
    """

    truth = _binary_array(labels, "labels")
    lengths = [stop - start + 1 for start, stop in _inclusive_events(truth)]
    if not lengths:
        raise ValueError("cannot derive an anomaly length without labelled events")
    return int(np.median(np.asarray(lengths, dtype=np.float64)))


def _inclusive_events(binary: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(np.diff(binary.astype(np.int8)) == 1) + 1
    stops = np.flatnonzero(np.diff(binary.astype(np.int8)) == -1)
    if len(stops) and (not len(starts) or stops[0] < starts[0]):
        starts = np.concatenate(([0], starts))
    if len(starts) and (not len(stops) or stops[-1] < starts[-1]):
        stops = np.concatenate((stops, [len(binary) - 1]))
    return list(zip(starts.astype(int).tolist(), stops.astype(int).tolist()))


def _merge_buffered_events(
    length: int, events: list[tuple[int, int]], buffer: int
) -> list[tuple[int, int]]:
    half = buffer // 2
    start = max(events[0][0] - half, 0)
    merged: list[tuple[int, int]] = []
    for current, following in zip(events[:-1], events[1:]):
        if current[1] + half < following[0] - half:
            merged.append((start, current[1] + half))
            start = following[0] - half
    merged.append((start, min(events[-1][1] + half, length - 1)))
    return merged


def _extend_labels(
    labels: np.ndarray, events: list[tuple[int, int]], buffer: int
) -> np.ndarray:
    extended = labels.astype(np.float64).copy()
    length = len(extended)
    if buffer == 0:
        return extended
    for start, stop in events:
        right = np.arange(stop + 1, min(stop + buffer // 2 + 1, length))
        extended[right] += np.sqrt(1.0 - (right - stop) / buffer)
        left = np.arange(max(start - buffer // 2, 0), start)
        extended[left] += np.sqrt(1.0 - (start - left) / buffer)
    return np.minimum(np.ones(length, dtype=np.float64), extended)


def _finite_array(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _binary_array(values: Sequence[bool] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if np.issubdtype(array.dtype, np.number):
        numeric = array.astype(np.float64)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{name} contains NaN or infinite values")
        if not np.all((numeric == 0.0) | (numeric == 1.0)):
            raise ValueError(f"{name} must contain only binary 0/1 decisions")
    elif array.dtype != np.bool_:
        raise ValueError(f"{name} must contain only binary 0/1 decisions")
    return array.astype(bool)
