from __future__ import annotations

import copy

import pytest

from experiments import run_external_denoiser_benchmark_v1 as frozen
from experiments import run_external_denoiser_benchmark_v1_recovery as recovery


def _methods() -> dict[str, dict[str, float | int]]:
    return copy.deepcopy(recovery.EXPECTED)


def test_recovery_changes_only_median_float_tolerance() -> None:
    methods = _methods()
    methods["median_filter_w3"]["overall_os_nrmse"] = 0.1548290545050969
    methods["median_filter_w3"]["anomaly_region_os_nrmse"] = 0.18997987258874352
    recovery.recovered_assert_existing_row_parity(methods)


def test_recovery_rejects_larger_median_difference() -> None:
    methods = _methods()
    methods["median_filter_w3"]["overall_os_nrmse"] = (
        float(methods["median_filter_w3"]["overall_os_nrmse"]) + 2.1e-9
    )
    with pytest.raises(frozen.BenchmarkError, match="recovered metric parity"):
        recovery.recovered_assert_existing_row_parity(methods)


def test_recovery_keeps_other_float_and_count_checks_strict() -> None:
    methods = _methods()
    methods["denoiseapt"]["overall_os_nrmse"] = (
        float(methods["denoiseapt"]["overall_os_nrmse"]) + 1.1e-12
    )
    with pytest.raises(frozen.BenchmarkError, match="recovered metric parity"):
        recovery.recovered_assert_existing_row_parity(methods)
    methods = _methods()
    methods["median_filter_w3"]["evidence_lost"] = 504
    with pytest.raises(frozen.BenchmarkError, match="recovered count parity"):
        recovery.recovered_assert_existing_row_parity(methods)


def test_recovery_contract_is_narrow() -> None:
    assert recovery.MEDIAN_ABS_TOL == 2e-9
    assert recovery.OTHER_ABS_TOL == 1e-12
    assert set(recovery.EXPECTED) == {
        "corrupted_input",
        "median_filter_w3",
        "denoiseapt",
    }


def test_legacy_audit_schema_is_distinct_and_validated() -> None:
    summary = {
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": {"outputs.npz": {"sha256": "artifact", "bytes": 1}},
    }
    audit = {
        "status": "AUDIT_PASS",
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": summary["artifacts"],
    }
    recovery._validate_legacy_audit(summary, audit)
    audit["summary_sha256"] = "different"
    with pytest.raises(frozen.BenchmarkError, match="legacy audit result differs"):
        recovery._validate_legacy_audit(summary, audit)


def test_recovered_evaluate_handles_full_summary_and_compact_audit(monkeypatch) -> None:
    summary = {
        "protocol_id": "protocol",
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": {},
    }
    audit = {
        "status": "AUDIT_PASS",
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": {},
    }
    monkeypatch.setattr(recovery, "_validate_recovery_receipt", lambda **_: {})
    calls = iter(((summary, 1, True), (audit, 1, True)))
    monkeypatch.setattr(recovery, "_with_recovered_parity", lambda _: next(calls))
    monkeypatch.setattr(
        recovery,
        "_write_recovery_audit",
        lambda canonical, legacy, **_: {"canonical": canonical, "legacy": legacy},
    )
    result = recovery.evaluate_recovered()
    assert result["summary"] is summary
    assert result["recovery_audit"]["legacy"] is audit


def test_finalize_after_commit_loads_canonical_summary(monkeypatch, tmp_path) -> None:
    summary = {
        "protocol_id": "protocol",
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": {},
    }
    audit = {
        "status": "AUDIT_PASS",
        "execution_identity_sha256": "identity",
        "summary_sha256": "summary",
        "artifacts": {},
    }
    monkeypatch.setattr(recovery, "AUDIT_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(recovery, "_validate_recovery_receipt", lambda **_: {})
    monkeypatch.setattr(frozen, "_load_json", lambda _: summary)
    monkeypatch.setattr(recovery, "_with_recovered_parity", lambda _: (audit, 1, True))
    monkeypatch.setattr(
        recovery,
        "_write_recovery_audit",
        lambda canonical, legacy, **_: {"canonical": canonical, "legacy": legacy},
    )
    result = recovery.finalize_recovery_audit()
    assert result["summary"] is summary
    assert result["legacy_audit"] is audit
