"""Dependency-light metrics for the DenoiseAPT demonstration.

The range-aware PR function in this module is intentionally named an
*approximation*.  It integrates soft-label PR-AUC across temporal tolerances,
but it is not a drop-in reimplementation of the official TSB-AD VUS-PR code.
Use the official TSB-AD evaluator for publication benchmark tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class VUSPRApproximation:
    """Transparent result for the demo's temporal-tolerance PR approximation."""

    value: float
    tolerances: tuple[int, ...]
    pr_auc_by_tolerance: tuple[float, ...]
    method: str = "soft-label temporal-tolerance average-precision mean (not official VUS-PR)"

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "tolerances": list(self.tolerances),
            "pr_auc_by_tolerance": list(self.pr_auc_by_tolerance),
            "method": self.method,
            "publication_note": "Recompute with the official TSB-AD VUS-PR implementation for reported results.",
        }


def rmse(reference: ArrayLike, candidate: ArrayLike) -> float:
    x, y = _paired_finite(reference, candidate)
    return float(np.sqrt(np.mean(np.square(x - y))))


def mae(reference: ArrayLike, candidate: ArrayLike) -> float:
    x, y = _paired_finite(reference, candidate)
    return float(np.mean(np.abs(x - y)))


def derivative_rmse(reference: ArrayLike, candidate: ArrayLike) -> float:
    x, y = _paired_finite(reference, candidate)
    return float(np.sqrt(np.mean(np.square(np.diff(x) - np.diff(y)))))


def snr_db(reference: ArrayLike, candidate: ArrayLike) -> float:
    """Return signal-to-error ratio in dB using reference power."""

    x, y = _paired_finite(reference, candidate)
    signal_power = float(np.mean(np.square(x)))
    error_power = float(np.mean(np.square(x - y)))
    if error_power == 0.0:
        return math.inf
    if signal_power == 0.0:
        return -math.inf
    return float(10.0 * np.log10(signal_power / error_power))


def signal_metrics(reference: ArrayLike, candidate: ArrayLike, noisy: ArrayLike | None = None) -> dict[str, float]:
    """Compute waveform fidelity metrics; include SNR improvement when possible."""

    result = {
        "rmse": rmse(reference, candidate),
        "mae": mae(reference, candidate),
        "snr_db": snr_db(reference, candidate),
        "derivative_rmse": derivative_rmse(reference, candidate),
    }
    if noisy is not None:
        input_snr = snr_db(reference, noisy)
        output_snr = result["snr_db"]
        result["input_snr_db"] = input_snr
        result["snr_improvement_db"] = output_snr - input_snr
    return result


def event_intervals(labels: Sequence[bool] | np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous positive intervals as half-open ``(start, stop)`` pairs."""

    binary = _binary_1d(labels, "labels")
    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def event_recall(
    labels: Sequence[bool] | np.ndarray,
    scores_or_predictions: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Fraction of labelled events containing at least one positive prediction."""

    truth = _binary_1d(labels, "labels")
    predicted = _threshold(scores_or_predictions, threshold, len(truth))
    intervals = event_intervals(truth)
    if not intervals:
        return 0.0
    detected = sum(bool(np.any(predicted[start:stop])) for start, stop in intervals)
    return float(detected / len(intervals))


def anomaly_erasure_rate(
    labels: Sequence[bool] | np.ndarray,
    before_scores_or_predictions: ArrayLike,
    after_scores_or_predictions: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Rate of previously detectable genuine events lost after denoising.

    The denominator is the number of labelled events detected in the observed
    (before-denoising) signal.  Returning zero when none were initially
    detectable avoids claiming erasure where the detector had no evidence.
    """

    truth = _binary_1d(labels, "labels")
    before = _threshold(before_scores_or_predictions, threshold, len(truth))
    after = _threshold(after_scores_or_predictions, threshold, len(truth))
    before_detected = []
    erased = 0
    for start, stop in event_intervals(truth):
        detectable = bool(np.any(before[start:stop]))
        before_detected.append(detectable)
        if detectable and not np.any(after[start:stop]):
            erased += 1
    denominator = sum(before_detected)
    return float(erased / denominator) if denominator else 0.0


def false_event_generation_rate(
    labels: Sequence[bool] | np.ndarray,
    before_scores_or_predictions: ArrayLike,
    after_scores_or_predictions: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """Fraction of after-denoising predicted events that are newly false.

    An after-event is newly false when it overlaps neither a labelled anomaly
    nor any before-denoising predicted event.  The denominator is all predicted
    after-events, making the metric bounded and interpretable per window.
    """

    truth = _binary_1d(labels, "labels")
    before = _threshold(before_scores_or_predictions, threshold, len(truth))
    after = _threshold(after_scores_or_predictions, threshold, len(truth))
    after_events = event_intervals(after)
    if not after_events:
        return 0.0
    generated = sum(
        not np.any(truth[start:stop]) and not np.any(before[start:stop])
        for start, stop in after_events
    )
    return float(generated / len(after_events))


def precision_recall_auc(labels: Sequence[bool] | np.ndarray, scores: ArrayLike) -> float:
    """Average precision with tied scores evaluated as one threshold group."""

    truth = _binary_1d(labels, "labels").astype(np.float64)
    values = _finite_1d(scores, "scores")
    if len(truth) != len(values):
        raise ValueError("labels and scores must have equal lengths")
    return _soft_average_precision(truth, values)


def vus_pr_approximation(
    labels: Sequence[bool] | np.ndarray,
    scores: ArrayLike,
    *,
    max_tolerance: int | None = None,
    tolerance_steps: int = 11,
) -> VUSPRApproximation:
    """Approximate range-aware VUS-PR by averaging over temporal tolerances.

    At each tolerance, labels inside true events retain weight one.  The weight
    decays linearly in the buffer on both sides of each event.  Average
    precision is computed with weighted positives, then averaged over the
    requested tolerances.  This captures the central temporal-tolerance idea of
    VUS-PR, but omits the official implementation's range existence reward and
    should therefore be labelled ``VUS-PR approximation`` in the interface.
    """

    truth = _binary_1d(labels, "labels")
    values = _finite_1d(scores, "scores")
    if len(truth) != len(values):
        raise ValueError("labels and scores must have equal lengths")
    if tolerance_steps < 1:
        raise ValueError("tolerance_steps must be positive")
    if max_tolerance is None:
        lengths = [stop - start for start, stop in event_intervals(truth)]
        max_tolerance = int(round(float(np.median(lengths)))) if lengths else 0
    if max_tolerance < 0:
        raise ValueError("max_tolerance must be non-negative")
    count = min(tolerance_steps, max_tolerance + 1)
    tolerances = tuple(sorted(set(np.rint(np.linspace(0, max_tolerance, count)).astype(int).tolist())))
    aucs = tuple(_soft_average_precision(_soft_labels(truth, tolerance), values) for tolerance in tolerances)
    value = float(np.mean(aucs)) if aucs else 0.0
    return VUSPRApproximation(value, tolerances, aucs)


def demo_metrics(
    reference: ArrayLike,
    observation: ArrayLike,
    candidate: ArrayLike,
    labels: Sequence[bool] | np.ndarray,
    before_scores: ArrayLike,
    after_scores: ArrayLike,
    *,
    threshold: float = 0.5,
    max_tolerance: int | None = None,
) -> dict[str, object]:
    """Return the metric bundle consumed by the interactive dashboard."""

    waveform = signal_metrics(reference, candidate, observation)
    approximation = vus_pr_approximation(labels, after_scores, max_tolerance=max_tolerance)
    return {
        "signal": waveform,
        "anomaly": {
            "event_recall": event_recall(labels, after_scores, threshold),
            "anomaly_erasure_rate": anomaly_erasure_rate(labels, before_scores, after_scores, threshold),
            "false_event_generation_rate": false_event_generation_rate(
                labels, before_scores, after_scores, threshold
            ),
            "auprc": precision_recall_auc(labels, after_scores),
            "vus_pr_approximation": approximation.to_dict(),
        },
    }


def _soft_labels(labels: np.ndarray, tolerance: int) -> np.ndarray:
    soft = labels.astype(np.float64)
    if tolerance == 0:
        return soft
    length = len(labels)
    for start, stop in event_intervals(labels):
        for distance in range(1, tolerance + 1):
            weight = 1.0 - distance / (tolerance + 1.0)
            left = start - distance
            right = stop - 1 + distance
            if left >= 0:
                soft[left] = max(soft[left], weight)
            if right < length:
                soft[right] = max(soft[right], weight)
    return soft


def _soft_average_precision(soft_truth: np.ndarray, scores: np.ndarray) -> float:
    total_positive = float(np.sum(soft_truth))
    if total_positive <= 0.0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = soft_truth[order]
    cumulative_tp = np.cumsum(sorted_truth)
    cumulative_count = np.arange(1, len(scores) + 1, dtype=np.float64)
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores) - 1]
    recall = cumulative_tp[threshold_ends] / total_positive
    precision = cumulative_tp[threshold_ends] / cumulative_count[threshold_ends]
    recall_delta = np.diff(np.r_[0.0, recall])
    return float(np.clip(np.sum(recall_delta * precision), 0.0, 1.0))


def _paired_finite(left: ArrayLike, right: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = _finite_1d(left, "reference")
    y = _finite_1d(right, "candidate")
    if len(x) != len(y):
        raise ValueError("reference and candidate must have equal lengths")
    return x, y


def _finite_1d(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _binary_1d(values: Sequence[bool] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array.astype(float))):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array.astype(bool)


def _threshold(values: ArrayLike, threshold: float, expected_length: int) -> np.ndarray:
    scores = _finite_1d(values, "scores_or_predictions")
    if len(scores) != expected_length:
        raise ValueError("labels and scores_or_predictions must have equal lengths")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    return scores >= threshold
