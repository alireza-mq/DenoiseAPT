from __future__ import annotations

import numpy as np
import pytest

from denoiseapt.experiment_statistics import (
    holm_adjust,
    paired_cluster_bootstrap,
    paired_sign_permutation_test,
    preservation_claim_gate,
)


def test_bootstrap_is_paired_reproducible_and_cluster_counted() -> None:
    baseline = {f"s{i}": float(i + 2) for i in range(12)}
    candidate = {key: value - 0.5 for key, value in baseline.items()}
    first = paired_cluster_bootstrap(baseline, candidate, replicates=500, seed=7)
    second = paired_cluster_bootstrap(baseline, candidate, replicates=500, seed=7)
    assert first == second
    assert first.clusters == 12
    assert first.estimate == pytest.approx(-0.5)
    assert first.upper < 0


def test_relative_bootstrap_omits_zero_denominator() -> None:
    result = paired_cluster_bootstrap(
        [0.0, 1.0, 2.0], [1.0, 1.05, 2.10],
        relative_to_left=True, replicates=500, seed=9,
    )
    assert result.clusters == 2
    assert result.estimate == pytest.approx(0.05)


def test_sign_permutation_detects_consistent_reduction() -> None:
    left = np.arange(1.0, 21.0)
    right = left - 0.4
    p_value = paired_sign_permutation_test(
        left, right, alternative="less", replicates=20_000, seed=3
    )
    assert p_value < 0.001


def test_holm_adjust_preserves_original_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.04, 0.001, 0.03])
    assert adjusted == pytest.approx([0.06, 0.003, 0.06])


def test_claim_gate_passes_only_joint_requirements() -> None:
    series = {f"s{i}": 0.50 + i * 0.002 for i in range(20)}
    dapt = {key: value - 0.10 for key, value in series.items()}
    rmse = {key: 1.0 + i * 0.01 for i, key in enumerate(series)}
    dapt_rmse = {key: value * 1.01 for key, value in rmse.items()}
    gate = preservation_claim_gate(
        series, dapt, rmse, dapt_rmse,
        transfer_cgan_erasure=series,
        transfer_denoiseapt_erasure=dapt,
        replicates=1_000,
        seed=11,
    )
    assert gate.supported
    failed = preservation_claim_gate(
        series,
        dapt,
        rmse,
        {key: value * 1.20 for key, value in rmse.items()},
        replicates=1_000,
        seed=11,
    )
    assert not failed.supported
    assert not failed.restoration_noninferiority_pass


def test_mapping_pairs_must_match() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        paired_cluster_bootstrap({"a": 1.0}, {"b": 1.0})
