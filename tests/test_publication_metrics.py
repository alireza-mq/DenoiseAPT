from __future__ import annotations

import numpy as np
import pytest

from denoiseapt.publication_metrics import (
    TSB_AD_VUS_PROVENANCE,
    event_decision_metrics,
    median_anomaly_length,
    official_vus_pr,
)


def _golden_case() -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros(64, dtype=int)
    labels[10:15] = 1
    labels[38:44] = 1
    scores = np.linspace(0.02, 0.18, 64)
    scores[9:16] = np.array([0.25, 0.55, 0.72, 0.91, 0.84, 0.63, 0.31])
    scores[37:45] = np.array([0.22, 0.48, 0.68, 0.93, 0.87, 0.79, 0.56, 0.29])
    scores[53] = 0.62
    return labels, scores


def test_official_vus_pr_matches_pinned_tsb_ad_golden_value() -> None:
    """Golden value generated directly by pinned RangeAUC_volume_opt()."""

    labels, scores = _golden_case()
    result = official_vus_pr(labels, scores, max_buffer=11, threshold_count=250)
    assert result.value == pytest.approx(0.9750688705234158, abs=1e-15)
    assert len(result.pr_auc_by_buffer) == 12
    assert result.max_buffer == 11
    assert result.threshold_count == 250
    assert result.to_dict()["metric"] == "VUS-PR"


def test_official_vus_pr_records_exact_source_identity() -> None:
    provenance = TSB_AD_VUS_PROVENANCE
    assert provenance.package == "TSB-AD"
    assert provenance.package_version == "1.5"
    assert provenance.commit == "e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48"
    assert provenance.source_sha256 == (
        "1fcddedf5ada1d5221f39ee568c7fddb9e7181bd7e1c19636c2cbecf40c97707"
    )
    assert provenance.wrapper_sha256 == (
        "13957d15d3ebb2b743a4bebfa3a68af31e82a58ecc568557b4081ad90e38094d"
    )
    assert provenance.license == "Apache-2.0"


def test_official_vus_pr_has_no_invalid_or_approximate_fallback() -> None:
    labels = np.zeros(16, dtype=int)
    scores = np.linspace(0.0, 1.0, 16)
    with pytest.raises(ValueError, match="undefined"):
        official_vus_pr(labels, scores, max_buffer=3)
    with pytest.raises(ValueError, match="max_buffer"):
        official_vus_pr(np.r_[0, 1, np.zeros(14)], scores, max_buffer=16)


def test_median_anomaly_length_is_an_explicit_policy_helper() -> None:
    labels = np.zeros(30, dtype=int)
    labels[2:5] = 1
    labels[10:17] = 1
    labels[22:27] = 1
    assert median_anomaly_length(labels) == 5


def test_event_decision_metrics_count_all_four_transitions() -> None:
    labels = np.zeros(22, dtype=int)
    labels[2:4] = 1
    labels[8:10] = 1
    labels[14:16] = 1

    reference = np.zeros(22, dtype=int)
    reference[[2, 8]] = 1
    observation = np.zeros(22, dtype=int)
    observation[[3, 14]] = 1
    candidate = np.zeros(22, dtype=int)
    candidate[[2, 8, 19]] = 1

    result = event_decision_metrics(labels, reference, observation, candidate)
    assert result.labelled_events == 3
    assert result.reference_detectable_events == 2
    assert result.observation_detectable_events == 2
    assert result.candidate_detectable_events == 2
    assert result.reference_retention.numerator == 2
    assert result.reference_retention.denominator == 2
    assert result.reference_retention.rate == 1.0
    assert result.denoising_erasure.numerator == 1
    assert result.denoising_erasure.rate == 0.5
    assert result.denoising_recovery.numerator == 1
    assert result.denoising_recovery.rate == 1.0
    assert result.false_event_generation.numerator == 1
    assert result.false_event_generation.denominator == 3
    assert result.false_event_generation.rate == pytest.approx(1 / 3)


def test_event_decision_zero_denominators_are_not_applicable() -> None:
    labels = np.zeros(12, dtype=int)
    labels[3:5] = 1
    zeros = np.zeros(12, dtype=int)
    result = event_decision_metrics(labels, zeros, zeros, zeros)
    assert result.reference_retention.denominator == 0
    assert result.reference_retention.rate is None
    assert result.denoising_erasure.rate is None
    assert result.denoising_recovery.rate is None
    assert result.false_event_generation.rate is None
    assert result.to_dict()["reference_retention"]["status"] == "not_applicable"


def test_event_decisions_reject_continuous_scores() -> None:
    labels = np.array([0, 1, 0, 0], dtype=int)
    with pytest.raises(ValueError, match="binary"):
        event_decision_metrics(labels, [0.1, 0.9, 0.2, 0.1], labels, labels)
