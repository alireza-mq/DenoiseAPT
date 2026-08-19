from __future__ import annotations

import copy

import numpy as np
import pytest

from denoiseapt.evidence_controller import (
    ControllerConfig,
    EvidencePreservationController,
    WitnessSpec,
    array_sha256,
    frozen_config_payload,
    load_frozen_config,
)


def _abs_score(signal: np.ndarray) -> np.ndarray:
    return np.abs(signal).astype(np.float32)


def _witness(
    witness_id: str = "A", *, threshold: float = 1.0, scorer=_abs_score
) -> WitnessSpec:
    return WitnessSpec(
        witness_id=witness_id,
        scorer=scorer,
        threshold=threshold,
        model_sha256=f"model-{witness_id}",
        threshold_source="development-validation-only",
        threshold_source_sha256=f"threshold-{witness_id}",
    )


def _config(**changes) -> ControllerConfig:
    values = {
        "preserve_peak_ratio": 0.8,
        "fabrication_dilation": 0,
        "repair_padding": 0,
        "tile_width": 1,
        "blend_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
        "max_auto_changed_fraction": 1.0,
    }
    values.update(changes)
    return ControllerConfig(**values)


def test_passing_candidate_is_accepted_without_intervention() -> None:
    observation = np.zeros(16, dtype=np.float32)
    observation[6] = 2.0
    candidate = observation.copy()
    controller = EvidencePreservationController([_witness()], _config())

    result = controller.project(observation, candidate)

    assert result.decision == "accept"
    assert result.auto_committed
    assert result.certificate.passed
    assert np.array_equal(result.automatic_signal, candidate)
    assert np.count_nonzero(result.beta) == 0


def test_erased_event_gets_grid_minimal_local_blend() -> None:
    observation = np.zeros(16, dtype=np.float32)
    observation[6] = 2.0
    candidate = np.zeros_like(observation)
    candidate[6] = 0.4
    controller = EvidencePreservationController([_witness()], _config())

    result = controller.project(observation, candidate)

    assert result.decision == "blend"
    assert result.certificate.passed
    assert result.beta[6] == pytest.approx(0.75)
    assert np.count_nonzero(result.beta) == 1
    assert result.automatic_signal[6] == pytest.approx(1.6)
    # The next smaller lattice value does not retain the required 80% peak.
    lower = candidate.copy()
    lower[6] = candidate[6] + 0.5 * (observation[6] - candidate[6])
    assert not controller.verify(observation, lower).passed


def test_retention_uses_original_component_not_timing_dilation() -> None:
    observation = np.zeros(12, dtype=np.float32)
    observation[5] = 2.0
    candidate = np.zeros_like(observation)
    candidate[6] = 2.0  # Within fabrication tolerance but outside original component.
    controller = EvidencePreservationController(
        [_witness()], _config(fabrication_dilation=2)
    )

    before = controller.verify(observation, candidate)
    result = controller.project(observation, candidate)

    assert not before.witnesses[0].preservation_passed
    assert before.witnesses[0].fabrication_passed
    assert result.certificate.passed
    assert result.automatic_signal[5] >= 1.6 - 1e-6


def test_candidate_only_threshold_event_is_suppressed() -> None:
    observation = np.zeros(16, dtype=np.float32)
    candidate = np.zeros_like(observation)
    candidate[11] = 2.0
    controller = EvidencePreservationController([_witness()], _config())

    result = controller.project(observation, candidate)

    assert result.certificate.passed
    assert result.decision == "blend"
    assert result.beta[11] == pytest.approx(0.75)
    assert result.automatic_signal[11] < 1.0


def test_strict_union_requires_every_configured_witness() -> None:
    observation = np.zeros(10, dtype=np.float32)
    observation[4] = 2.0
    candidate = observation.copy()

    def inverse_score(signal: np.ndarray) -> np.ndarray:
        return np.abs(2.0 - signal).astype(np.float32)

    config = _config(
        required_witness_ids=("A", "B"),
        preserve_peak_ratio=1.0,
    )
    controller = EvidencePreservationController(
        [_witness("A"), _witness("B", scorer=inverse_score)], config
    )
    result = controller.project(observation, candidate)

    assert {w.witness_id for w in result.certificate.witnesses} == {"A", "B"}
    assert result.certificate.passed
    with pytest.raises(ValueError, match="Strict-union witness set"):
        EvidencePreservationController([_witness("A")], config)


def test_intervention_budget_causes_exact_observation_abstention() -> None:
    observation = np.zeros(16, dtype=np.float32)
    observation[5] = 2.0
    candidate = np.zeros_like(observation)
    controller = EvidencePreservationController(
        [_witness()], _config(max_auto_changed_fraction=0.01)
    )

    result = controller.project(observation, candidate)

    assert result.decision == "abstain"
    assert not result.auto_committed
    assert result.fallback_reason == "changed_fraction_budget_exceeded"
    assert np.array_equal(result.automatic_signal, observation)
    assert np.array_equal(result.beta, np.ones_like(observation))
    assert result.certificate.passed


def test_nondeterministic_witness_abstains_unverified() -> None:
    calls = {"count": 0}

    def changing_score(signal: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        return np.abs(signal) + np.float32(calls["count"] * 1e-3)

    observation = np.zeros(8, dtype=np.float32)
    candidate = np.ones(8, dtype=np.float32)
    controller = EvidencePreservationController(
        [_witness(scorer=changing_score)], _config()
    )

    result = controller.project(observation, candidate)

    assert result.decision == "abstain"
    assert result.certificate.status == "unverified"
    assert np.array_equal(result.automatic_signal, observation)
    assert "constraint_build_failed" in (result.fallback_reason or "")


def test_concern_can_trigger_preemptive_but_noncertifying_blend() -> None:
    observation = np.ones(12, dtype=np.float32) * 0.4
    candidate = np.zeros_like(observation)
    concern = np.zeros_like(observation)
    concern[4:6] = 0.9
    controller = EvidencePreservationController(
        [_witness(threshold=10.0)],
        _config(concern_min_beta=0.25, concern_threshold=0.6),
    )

    result = controller.project(observation, candidate, concern=concern)

    assert result.decision == "blend"
    assert np.all(result.beta[4:6] == 0.25)
    assert result.certificate.passed
    assert any("concern:preemptive" in interval.reasons for interval in result.intervals)


def test_decision_audit_is_deterministic_and_latency_is_separate() -> None:
    observation = np.zeros(16, dtype=np.float32)
    observation[6] = 2.0
    candidate = np.zeros_like(observation)
    controller = EvidencePreservationController([_witness()], _config())
    provenance = {"model_seed": 17, "checkpoint_sha256": "generator-checkpoint"}

    first = controller.project(
        observation, candidate, base_provenance=provenance, generator_latency_ms=3.5
    )
    second = controller.project(
        observation, candidate, base_provenance=provenance, generator_latency_ms=3.5
    )

    assert first.audit.decision_hash == second.audit.decision_hash
    assert np.array_equal(first.automatic_signal, second.automatic_signal)
    assert first.provenance["base"] == provenance
    assert first.generator_latency_ms == 3.5
    assert first.total_latency_ms == pytest.approx(3.5 + first.controller_latency_ms)
    assert "controller_latency_ms" not in str(first.audit.to_dict())


def test_human_override_is_reversible_and_reverified() -> None:
    observation = np.zeros(16, dtype=np.float32)
    observation[6] = 2.0
    candidate = np.zeros_like(observation)
    controller = EvidencePreservationController([_witness()], _config())
    automatic = controller.project(observation, candidate)
    session = controller.new_session(observation, candidate, automatic)

    changed, certificate = session.apply(
        "accept_candidate", 6, 7, expected_revision=0
    )
    assert changed[6] == 0.0
    assert certificate.status == "overridden"
    assert not certificate.passed

    restored, certificate = session.revert(expected_revision=1)
    assert np.array_equal(restored, automatic.automatic_signal)
    assert certificate.passed
    assert session.records[1].previous_record_hash == session.records[0].record_hash

    with pytest.raises(RuntimeError, match="Stale intervention revision"):
        session.apply("protect", 0, 1, expected_revision=0)


def test_frozen_config_artifact_detects_tampering() -> None:
    config = _config(required_witness_ids=("A", "B"))
    artifact = frozen_config_payload(
        config,
        development_metadata={
            "development_only": True,
            "spent_splits": ["validation", "test", "SMAP"],
            "confirmation_data_accessed": False,
        },
    )

    loaded = load_frozen_config(artifact)
    assert loaded.sha256 == config.sha256
    tampered = copy.deepcopy(artifact)
    tampered["controller_config"]["preserve_peak_ratio"] = 0.5
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        load_frozen_config(tampered)


def test_array_hash_includes_shape_and_dtype() -> None:
    values = np.array([1.0, 2.0], dtype=np.float32)
    assert array_sha256(values) != array_sha256(values.astype(np.float64))

