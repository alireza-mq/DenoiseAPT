"""Local preservation-concern cues for DenoiseAPT inference.

The resulting score is an inspection aid, not a calibrated probability of an
anomaly-preservation failure.  Signed scorer changes are retained separately so
the interface can distinguish possible evidence suppression from possible new
evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .corruptions import robust_signal_scale


@dataclass(frozen=True)
class ConcernConfig:
    """Weights, validation scales, and display thresholds for concern cues."""

    gamma_score: float = 0.50
    gamma_morphology: float = 0.30
    gamma_uncertainty: float = 0.20
    score_scale: float = 0.30
    morphology_scale: float = 1.00
    uncertainty_scale: float = 0.12
    medium_threshold: float = 0.32
    high_threshold: float = 0.62
    extrema_match_radius: int = 8
    smoothing_window: int = 7

    def __post_init__(self) -> None:
        weights = (self.gamma_score, self.gamma_morphology, self.gamma_uncertainty)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Concern weights must be non-negative with a positive sum.")
        if min(self.score_scale, self.morphology_scale, self.uncertainty_scale) <= 0:
            raise ValueError("Concern calibration scales must be positive.")
        if not 0 <= self.medium_threshold < self.high_threshold <= 1:
            raise ValueError("Concern thresholds must satisfy 0 <= medium < high <= 1.")
        if self.extrema_match_radius < 1:
            raise ValueError("extrema_match_radius must be positive.")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be positive.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ConcernConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConcernInterval:
    """One contiguous medium- or high-concern display interval."""

    start: int
    end: int
    level: str
    peak_concern: float
    mean_concern: float
    dominant_cue: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConcernCues:
    """Timestamp-level inspection cues and their interval summaries."""

    score_delta: NDArray[np.float32]
    possible_suppression: NDArray[np.float32]
    possible_emergence: NDArray[np.float32]
    morphology: NDArray[np.float32]
    uncertainty: NDArray[np.float32]
    concern: NDArray[np.float32]
    level: NDArray[np.str_]
    intervals: tuple[ConcernInterval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_delta": self.score_delta.tolist(),
            "possible_suppression": self.possible_suppression.tolist(),
            "possible_emergence": self.possible_emergence.tolist(),
            "morphology": self.morphology.tolist(),
            "uncertainty": self.uncertainty.tolist(),
            "concern": self.concern.tolist(),
            "level": self.level.tolist(),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "interpretation": (
                "Inspection cues only; concern levels are not calibrated failure probabilities."
            ),
        }


def compute_concern_cues(
    observation: ArrayLike,
    candidate: ArrayLike,
    stochastic_candidates: ArrayLike,
    observation_scores: ArrayLike,
    candidate_scores: ArrayLike,
    config: ConcernConfig | None = None,
) -> ConcernCues:
    """Compute local concern cues for a candidate restoration."""

    settings = config or ConcernConfig()
    observed = _signal(observation, "observation")
    restored = _signal(candidate, "candidate")
    scores_before = _signal(observation_scores, "observation_scores")
    scores_after = _signal(candidate_scores, "candidate_scores")
    samples = np.asarray(stochastic_candidates, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError("stochastic_candidates must have shape [passes, time].")
    expected_length = observed.size
    if any(array.size != expected_length for array in (restored, scores_before, scores_after)):
        raise ValueError("All concern inputs must use the same time dimension.")
    if samples.shape[1] != expected_length:
        raise ValueError("Stochastic candidate length does not match the observation.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Stochastic candidates must contain only finite values.")

    score_delta = (scores_after - scores_before).astype(np.float32)
    score_magnitude = np.clip(np.abs(score_delta) / settings.score_scale, 0.0, 1.0)
    suppression = np.clip(-score_delta / settings.score_scale, 0.0, 1.0)
    emergence = np.clip(score_delta / settings.score_scale, 0.0, 1.0)

    morphology_raw = morphology_change_cue(
        observed, restored, match_radius=settings.extrema_match_radius
    )
    morphology = np.clip(
        morphology_raw / settings.morphology_scale, 0.0, 1.0
    ).astype(np.float32)

    # Median absolute deviation captures disagreement without one stochastic
    # pass dominating the display cue.
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median[None, :]), axis=0)
    uncertainty = np.clip(mad / settings.uncertainty_scale, 0.0, 1.0).astype(
        np.float32
    )

    weight_sum = settings.gamma_score + settings.gamma_morphology + settings.gamma_uncertainty
    concern = (
        settings.gamma_score * score_magnitude
        + settings.gamma_morphology * morphology
        + settings.gamma_uncertainty * uncertainty
    ) / weight_sum
    concern = _moving_average(concern, settings.smoothing_window)
    concern = np.clip(concern, 0.0, 1.0).astype(np.float32)
    level = np.full(expected_length, "low", dtype="<U6")
    level[concern >= settings.medium_threshold] = "medium"
    level[concern >= settings.high_threshold] = "high"

    intervals = tuple(
        _summarize_intervals(
            concern,
            level,
            score_magnitude.astype(np.float32),
            morphology,
            uncertainty,
        )
    )
    return ConcernCues(
        score_delta=score_delta,
        possible_suppression=suppression.astype(np.float32),
        possible_emergence=emergence.astype(np.float32),
        morphology=morphology,
        uncertainty=uncertainty,
        concern=concern,
        level=level,
        intervals=intervals,
    )


def morphology_change_cue(
    observation: ArrayLike, candidate: ArrayLike, *, match_radius: int = 8
) -> NDArray[np.float32]:
    """Measure local amplitude and timing changes around matched extrema."""

    observed = _signal(observation, "observation")
    restored = _signal(candidate, "candidate")
    if observed.size != restored.size:
        raise ValueError("Observation and candidate lengths must match.")
    if match_radius < 1:
        raise ValueError("match_radius must be positive.")

    scale = robust_signal_scale(observed)
    # A light smoothing pass prevents individual high-frequency noise samples
    # from being interpreted as separate morphological events.
    smoothed_observed = _moving_average(observed, 5)
    smoothed_restored = _moving_average(restored, 5)
    original_extrema = _local_extrema(smoothed_observed, scale)
    restored_extrema = _local_extrema(smoothed_restored, scale)

    cue = 0.20 * np.clip(np.abs(restored - observed) / scale, 0.0, 1.0)
    matched_restored: set[int] = set()
    for original_index, polarity in original_extrema:
        matches = [
            (abs(restored_index - original_index), restored_index)
            for restored_index, restored_polarity in restored_extrema
            if restored_polarity == polarity
            and abs(restored_index - original_index) <= match_radius
        ]
        if matches:
            distance, restored_index = min(matches)
            matched_restored.add(restored_index)
            amplitude_error = min(
                abs(float(restored[restored_index] - observed[original_index])) / scale,
                1.0,
            )
            event_change = 0.55 * amplitude_error + 0.45 * distance / match_radius
        else:
            event_change = 1.0
        _spread_peak(cue, original_index, float(event_change), match_radius)

    # Candidate-only extrema are a separate cue for possible event fabrication.
    for restored_index, _ in restored_extrema:
        if restored_index not in matched_restored:
            nearby_original = any(
                abs(original_index - restored_index) <= match_radius
                for original_index, _ in original_extrema
            )
            if not nearby_original:
                _spread_peak(cue, restored_index, 0.8, match_radius)

    return np.clip(cue, 0.0, 1.0).astype(np.float32)


def _local_extrema(signal: NDArray[np.float32], scale: float) -> list[tuple[int, int]]:
    if signal.size < 3:
        return []
    left_change = signal[1:-1] - signal[:-2]
    right_change = signal[2:] - signal[1:-1]
    prominence_floor = 0.01 * scale
    maxima = (left_change > prominence_floor) & (right_change <= -prominence_floor)
    minima = (left_change < -prominence_floor) & (right_change >= prominence_floor)
    result = [(int(index + 1), 1) for index in np.flatnonzero(maxima)]
    result.extend((int(index + 1), -1) for index in np.flatnonzero(minima))
    result.sort()
    return result


def _spread_peak(
    target: NDArray[np.floating[Any]], center: int, height: float, radius: int
) -> None:
    start = max(0, center - radius)
    end = min(target.size, center + radius + 1)
    positions = np.arange(start, end)
    weights = 1.0 - np.abs(positions - center) / (radius + 1)
    target[start:end] = np.maximum(target[start:end], height * weights)


def _summarize_intervals(
    concern: NDArray[np.float32],
    levels: NDArray[np.str_],
    score: NDArray[np.float32],
    morphology: NDArray[np.float32],
    uncertainty: NDArray[np.float32],
) -> Sequence[ConcernInterval]:
    intervals: list[ConcernInterval] = []
    active = levels != "low"
    padded = np.pad(active.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    cue_names = ("score change", "morphology", "stochastic disagreement")
    for start, end in zip(starts, ends):
        cue_means = (
            float(np.mean(score[start:end])),
            float(np.mean(morphology[start:end])),
            float(np.mean(uncertainty[start:end])),
        )
        interval_level = "high" if np.any(levels[start:end] == "high") else "medium"
        intervals.append(
            ConcernInterval(
                start=int(start),
                end=int(end),
                level=interval_level,
                peak_concern=float(np.max(concern[start:end])),
                mean_concern=float(np.mean(concern[start:end])),
                dominant_cue=cue_names[int(np.argmax(cue_means))],
            )
        )
    return intervals


def _moving_average(values: ArrayLike, window: int) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if window <= 1:
        return array.copy()
    window = min(int(window), array.size)
    if window % 2 == 0 and window > 1:
        window -= 1
    pad = window // 2
    padded = np.pad(array, (pad, pad), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _signal(values: ArrayLike, name: str) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < 8 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain at least eight finite values.")
    return array
