from __future__ import annotations

import math

import numpy as np
import pytest

from denoiseapt.metrics import (
    anomaly_erasure_rate,
    demo_metrics,
    derivative_rmse,
    event_intervals,
    event_recall,
    false_event_generation_rate,
    mae,
    precision_recall_auc,
    rmse,
    signal_metrics,
    snr_db,
    vus_pr_approximation,
)


def test_signal_metrics_exact_candidate() -> None:
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(reference, reference) == 0.0
    assert mae(reference, reference) == 0.0
    assert derivative_rmse(reference, reference) == 0.0
    assert math.isinf(snr_db(reference, reference))


def test_signal_metrics_improvement() -> None:
    reference = np.array([1.0, -1.0, 1.0, -1.0])
    noisy = np.array([2.0, -2.0, 2.0, -2.0])
    candidate = np.array([1.1, -1.1, 1.1, -1.1])
    metrics = signal_metrics(reference, candidate, noisy)
    assert metrics["snr_improvement_db"] > 0
    assert metrics["rmse"] == pytest.approx(0.1)


def test_event_intervals_and_recall() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=bool)
    assert event_intervals(labels) == [(1, 3), (5, 6), (7, 9)]
    predictions = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0])
    assert event_recall(labels, predictions) == pytest.approx(2 / 3)


def test_erasure_rate_counts_only_initially_detectable_events() -> None:
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    before = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    after = np.array([0, 1, 0, 0, 0, 0, 0, 0])
    assert anomaly_erasure_rate(labels, before, after) == pytest.approx(0.5)
    assert anomaly_erasure_rate(labels, np.zeros(8), after) == 0.0


def test_false_event_generation_rate() -> None:
    labels = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    before = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    after = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 0])
    assert false_event_generation_rate(labels, before, after) == pytest.approx(2 / 3)


def test_average_precision_perfect_and_reversed() -> None:
    labels = np.array([0, 1, 1, 0, 0], dtype=bool)
    perfect = np.array([0.0, 1.0, 0.9, 0.1, 0.2])
    reversed_scores = -perfect
    assert precision_recall_auc(labels, perfect) == pytest.approx(1.0)
    assert precision_recall_auc(labels, reversed_scores) < 1.0


def test_vus_pr_approximation_is_explicit_bounded_and_deterministic() -> None:
    labels = np.zeros(40, dtype=bool)
    labels[15:20] = True
    scores = np.zeros(40)
    scores[15:20] = [0.7, 0.8, 1.0, 0.9, 0.7]
    first = vus_pr_approximation(labels, scores, max_tolerance=5, tolerance_steps=6)
    second = vus_pr_approximation(labels, scores, max_tolerance=5, tolerance_steps=6)
    assert first == second
    assert 0.0 <= first.value <= 1.0
    assert first.tolerances == (0, 1, 2, 3, 4, 5)
    assert "not official VUS-PR" in first.method
    assert first.value > vus_pr_approximation(labels, -scores, max_tolerance=5).value


def test_no_positive_labels_have_zero_pr() -> None:
    labels = np.zeros(8, dtype=bool)
    scores = np.linspace(0, 1, 8)
    assert precision_recall_auc(labels, scores) == 0.0
    assert vus_pr_approximation(labels, scores).value == 0.0


def test_demo_metric_bundle_uses_approximation_label() -> None:
    reference = np.sin(np.arange(32) / 4)
    observation = reference + 0.2
    candidate = reference + 0.05
    labels = np.zeros(32, dtype=bool)
    labels[12:16] = True
    before = labels.astype(float)
    after = labels.astype(float)
    bundle = demo_metrics(reference, observation, candidate, labels, before, after)
    approximation = bundle["anomaly"]["vus_pr_approximation"]
    assert "not official VUS-PR" in approximation["method"]
    assert bundle["signal"]["snr_improvement_db"] > 0

