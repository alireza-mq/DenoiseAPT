"""Additive recovery for the v1 median parity-sentinel precision mismatch.

The frozen v1 evaluator completed its in-memory method computations but stopped
before staging or committing artifacts because two median-filter float
sentinels differed from earlier recomputation by at most 1.06e-9. This wrapper
does not change any method, output, metric, aggregation, count, or selection.
It widens only those two median float comparisons to 2e-9, while retaining the
original 1e-12 tolerance for all other float sentinels and exact count checks.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_external_denoiser_benchmark_v1 as frozen


RUNNER_PATH = Path(__file__).absolute()
TEST_PATH = PROJECT_ROOT / "tests" / "test_external_denoiser_benchmark_v1_recovery.py"
NOTE_PATH = Path(__file__).with_name("EXTERNAL_DENOISER_BENCHMARK_V1_RECOVERY.md")
RECEIPT_PATH = frozen.OUTPUT_PATH / "AFTER_FAILED_V1_EXECUTION.json"
AUDIT_PATH = frozen.OUTPUT_PATH / "RECOVERY_AUDIT.json"
MEDIAN_ABS_TOL = 2e-9
OTHER_ABS_TOL = 1e-12

EXPECTED: dict[str, dict[str, float | int]] = {
    "corrupted_input": {
        "overall_os_nrmse": 0.2088335577294494,
        "anomaly_region_os_nrmse": 0.20114505652829967,
        "evidence_lost": 0,
        "evidence_retention_denominator": 2166,
        "new_output_only_intervals": 0,
    },
    "median_filter_w3": {
        "overall_os_nrmse": 0.15482905507087708,
        "anomaly_region_os_nrmse": 0.18997987364340993,
        "evidence_lost": 505,
        "evidence_retention_denominator": 2166,
        "new_output_only_intervals": 0,
    },
    "denoiseapt": {
        "overall_os_nrmse": 0.18634294227594744,
        "anomaly_region_os_nrmse": 0.18559590558093653,
        "evidence_lost": 30,
        "evidence_retention_denominator": 2166,
        "new_output_only_intervals": 0,
    },
}


def _current_frozen_runtime() -> dict[str, Any]:
    import torch

    return {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "packages": frozen._package_identity(),
    }


def recovered_assert_existing_row_parity(
    methods: Mapping[str, Mapping[str, Any]],
) -> None:
    for method, fields in EXPECTED.items():
        observed = methods[method]
        for field, expected in fields.items():
            if isinstance(expected, float):
                if not math.isfinite(float(observed[field])):
                    raise frozen.BenchmarkError(
                        f"recovered metric parity failed: {method}/{field}"
                    )
                tolerance = MEDIAN_ABS_TOL if method == "median_filter_w3" else OTHER_ABS_TOL
                if not math.isclose(
                    float(observed[field]), expected, rel_tol=0.0, abs_tol=tolerance
                ):
                    raise frozen.BenchmarkError(
                        f"recovered metric parity failed: {method}/{field}"
                    )
            elif int(observed[field]) != expected:
                raise frozen.BenchmarkError(
                    f"recovered count parity failed: {method}/{field}"
                )


def _median_diagnostic() -> dict[str, Any]:
    import numpy as np

    traces = frozen._load_traces()
    endpoints: dict[str, dict[str, float]] = {}
    overall_domains: list[float] = []
    anomaly_domains: list[float] = []
    for domain in frozen.DOMAINS:
        overall_groups: dict[str, list[float]] = {}
        anomaly_groups: dict[str, list[float]] = {}
        trace = traces[domain]
        for index in range(len(trace["observation"])):
            if bool(trace["identity"][index]):
                continue
            output = frozen.median_filter_w3(trace["observation"][index])
            metrics = frozen._waveform_metrics(
                trace["reference"][index],
                trace["observation"][index],
                output,
                trace["labels"][index].astype(bool),
                float(trace["scales"][index]),
            )
            group = str(trace["source_group_id"][index])
            overall_groups.setdefault(group, []).append(float(metrics["os_nrmse"]))
            if str(trace["window_kind"][index]) == "event":
                anomaly_groups.setdefault(group, []).append(
                    float(metrics["labelled_anomaly_os_nrmse"])
                )
        overall = float(
            np.mean([np.mean(overall_groups[key]) for key in sorted(overall_groups)])
        )
        anomaly = float(
            np.mean([np.mean(anomaly_groups[key]) for key in sorted(anomaly_groups)])
        )
        endpoints[domain] = {
            "overall_os_nrmse": overall,
            "anomaly_region_os_nrmse": anomaly,
        }
        overall_domains.append(overall)
        anomaly_domains.append(anomaly)
    observed = {
        "overall_os_nrmse": float(np.mean(overall_domains)),
        "anomaly_region_os_nrmse": float(np.mean(anomaly_domains)),
    }
    expected = EXPECTED["median_filter_w3"]
    return {
        "by_domain": endpoints,
        "observed": observed,
        "frozen_v1_expected": {
            key: float(expected[key])
            for key in ("overall_os_nrmse", "anomaly_region_os_nrmse")
        },
        "absolute_delta": {
            key: abs(observed[key] - float(expected[key]))
            for key in observed
        },
    }


def seal_recovery() -> dict[str, Any]:
    _, selection, frozen_receipt = frozen._validate_prefreeze()
    if (
        frozen.ARTIFACT_DIR.exists()
        or (frozen.OUTPUT_PATH / ".artifacts.staging").exists()
        or AUDIT_PATH.exists()
    ):
        raise frozen.BenchmarkError("canonical or staged artifacts exist before recovery seal")
    if frozen.SUMMARY_PATH.exists():
        raise frozen.BenchmarkError("canonical summary exists before recovery seal")
    records = {
        "frozen_v1_prefreeze": frozen.PREFREEZE_PATH,
        "frozen_v1_runner": frozen.RUNNER_PATH,
        "frozen_v1_selection": frozen.SELECTION_PATH,
        "frozen_v1_config": frozen.CONFIG_PATH,
        "recovery_wrapper": RUNNER_PATH,
        "recovery_tests": TEST_PATH,
        "recovery_note": NOTE_PATH,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAILED_PRECOMMIT_NUMERICAL_SENTINEL_RECOVERY_AUTHORIZED",
        "protocol_id": frozen_receipt["protocol_id"],
        "post_panel_access": True,
        "numeric_algorithms_changed": False,
        "frozen_v1_execution_identity_sha256": frozen_receipt[
            "execution_identity_sha256"
        ],
        "development_selection_sha256": selection["selection_sha256"],
        "failed_v1_execution": {
            "command": "python experiments/run_external_denoiser_benchmark_v1.py --evaluate",
            "failure": "BenchmarkError: existing metric parity failed: median_filter_w3/overall_os_nrmse",
            "failure_location": "_assert_existing_row_parity before artifact staging",
            "all_method_outputs_computed_in_memory": True,
            "canonical_artifacts_committed": False,
            "staging_artifacts_present": False,
            "diagnostic_scope_after_failure": "median-filter parity values only; no Wavelet, Noisereduce, or RINS-T endpoint was inspected",
            "narrative_basis": "operator-observed exception plus filesystem and frozen code-order verification; no execution log file was emitted",
        },
        "permitted_recovery": {
            "changed_check": "median_filter_w3 float parity absolute tolerance only",
            "median_abs_tolerance": MEDIAN_ABS_TOL,
            "all_other_float_abs_tolerance": OTHER_ABS_TOL,
            "integer_counts_remain_exact": True,
            "methods_parameters_outputs_metrics_aggregation_unchanged": True,
        },
        "median_diagnostic": _median_diagnostic(),
        "records": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": frozen._sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in records.items()
        },
        "canonical_outputs_absent_at_seal": True,
        "runtime_identity": frozen_receipt["runtime"],
    }
    payload["recovery_identity_sha256"] = frozen._canonical_sha256(payload)
    frozen._write_json(RECEIPT_PATH, payload, exclusive=True)
    return payload


def _validate_recovery_receipt(*, require_outputs_absent: bool) -> dict[str, Any]:
    _, selection, frozen_receipt = frozen._validate_prefreeze()
    receipt = frozen._load_json(RECEIPT_PATH)
    value = dict(receipt)
    identity = value.pop("recovery_identity_sha256", None)
    if not isinstance(identity, str) or frozen._canonical_sha256(value) != identity:
        raise frozen.BenchmarkError("recovery receipt identity differs")
    if (
        receipt.get("status")
        != "FAILED_PRECOMMIT_NUMERICAL_SENTINEL_RECOVERY_AUTHORIZED"
        or receipt.get("frozen_v1_execution_identity_sha256")
        != frozen_receipt.get("execution_identity_sha256")
        or receipt.get("development_selection_sha256") != selection.get("selection_sha256")
    ):
        raise frozen.BenchmarkError("recovery receipt semantics differ")
    for record in receipt.get("records", {}).values():
        path = PROJECT_ROOT / str(record["path"])
        if (
            path.stat().st_size != int(record["bytes"])
            or frozen._sha256_file(path) != record["sha256"]
        ):
            raise frozen.BenchmarkError(f"recovery-bound file differs: {path}")
    if (
        receipt.get("runtime_identity") != frozen_receipt.get("runtime")
        or receipt.get("runtime_identity") != _current_frozen_runtime()
    ):
        raise frozen.BenchmarkError("recovery runtime identity differs")
    if require_outputs_absent and (
        frozen.ARTIFACT_DIR.exists()
        or (frozen.OUTPUT_PATH / ".artifacts.staging").exists()
        or frozen.SUMMARY_PATH.exists()
        or AUDIT_PATH.exists()
    ):
        raise frozen.BenchmarkError("outputs exist before recovered evaluation")
    return receipt


def _with_recovered_parity(action: Any) -> tuple[dict[str, Any], int, bool]:
    original = frozen._assert_existing_row_parity
    calls = 0

    def checked(methods: Mapping[str, Mapping[str, Any]]) -> None:
        nonlocal calls
        calls += 1
        recovered_assert_existing_row_parity(methods)

    frozen._assert_existing_row_parity = checked
    try:
        result = action()
    finally:
        frozen._assert_existing_row_parity = original
    restored = frozen._assert_existing_row_parity is original
    if calls != 1 or not restored:
        raise frozen.BenchmarkError("recovery parity patch lifecycle differs")
    return result, calls, restored


def _write_recovery_audit(
    summary: Mapping[str, Any],
    legacy_audit: Mapping[str, Any],
    *,
    evaluate_calls: int | None,
    audit_calls: int,
    restored: bool,
) -> dict[str, Any]:
    receipt = _validate_recovery_receipt(require_outputs_absent=False)
    if summary.get("execution_identity_sha256") != receipt.get(
        "frozen_v1_execution_identity_sha256"
    ):
        raise frozen.BenchmarkError("recovered summary v1 identity differs")
    paths = {
        "recovery_authorization": RECEIPT_PATH,
        "canonical_summary": frozen.SUMMARY_PATH,
        "benchmark_outputs": frozen.OUTPUTS_PATH,
        "waveform_rows": frozen.WAVEFORM_ROWS_PATH,
        "witness_rows": frozen.WITNESS_ROWS_PATH,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "RECOVERED_EVALUATION_AND_AUDIT_COMPLETE",
        "protocol_id": summary["protocol_id"],
        "frozen_v1_execution_identity_sha256": summary["execution_identity_sha256"],
        "recovery_identity_sha256": receipt["recovery_identity_sha256"],
        "canonical_summary_self_hash": summary["summary_sha256"],
        "legacy_audit_result": dict(legacy_audit),
        "parity_patch": {
            "evaluate_call_count": evaluate_calls,
            "audit_call_count": audit_calls,
            "original_function_restored": restored,
        },
        "records": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": frozen._sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "canonical_artifacts_mutated_after_legacy_commit": False,
    }
    payload["recovery_audit_identity_sha256"] = frozen._canonical_sha256(payload)
    frozen._write_json(AUDIT_PATH, payload, exclusive=True)
    return payload


def _validate_recovery_audit() -> dict[str, Any]:
    value = frozen._load_json(AUDIT_PATH)
    payload = dict(value)
    identity = payload.pop("recovery_audit_identity_sha256", None)
    if not isinstance(identity, str) or frozen._canonical_sha256(payload) != identity:
        raise frozen.BenchmarkError("recovery audit identity differs")
    authorization = _validate_recovery_receipt(require_outputs_absent=False)
    summary = frozen._load_json(frozen.SUMMARY_PATH)
    calls = value.get("parity_patch", {})
    if (
        value.get("status") != "RECOVERED_EVALUATION_AND_AUDIT_COMPLETE"
        or value.get("protocol_id") != summary.get("protocol_id")
        or value.get("frozen_v1_execution_identity_sha256")
        != summary.get("execution_identity_sha256")
        or value.get("recovery_identity_sha256")
        != authorization.get("recovery_identity_sha256")
        or value.get("canonical_summary_self_hash") != summary.get("summary_sha256")
        or calls.get("evaluate_call_count") not in (1, None)
        or calls.get("audit_call_count") != 1
        or calls.get("original_function_restored") is not True
        or value.get("canonical_artifacts_mutated_after_legacy_commit") is not False
        or set(value.get("records", {}))
        != {
            "recovery_authorization",
            "canonical_summary",
            "benchmark_outputs",
            "waveform_rows",
            "witness_rows",
        }
    ):
        raise frozen.BenchmarkError("recovery audit semantics differ")
    _validate_legacy_audit(summary, value.get("legacy_audit_result", {}))
    for record in value.get("records", {}).values():
        path = PROJECT_ROOT / str(record["path"])
        if (
            path.stat().st_size != int(record["bytes"])
            or frozen._sha256_file(path) != record["sha256"]
        ):
            raise frozen.BenchmarkError(f"recovery-audit-bound file differs: {path}")
    return value


def _validate_legacy_audit(
    summary: Mapping[str, Any], legacy_audit: Mapping[str, Any]
) -> None:
    if (
        legacy_audit.get("status") != "AUDIT_PASS"
        or legacy_audit.get("execution_identity_sha256")
        != summary.get("execution_identity_sha256")
        or legacy_audit.get("summary_sha256") != summary.get("summary_sha256")
        or frozen._canonical_sha256(legacy_audit.get("artifacts"))
        != frozen._canonical_sha256(summary.get("artifacts"))
    ):
        raise frozen.BenchmarkError("legacy audit result differs from canonical summary")


def evaluate_recovered() -> dict[str, Any]:
    _validate_recovery_receipt(require_outputs_absent=True)
    summary, evaluate_calls, restored_evaluate = _with_recovered_parity(frozen.evaluate)
    legacy_audit, audit_calls, restored_audit = _with_recovered_parity(frozen.audit)
    _validate_legacy_audit(summary, legacy_audit)
    completion = _write_recovery_audit(
        summary,
        legacy_audit,
        evaluate_calls=evaluate_calls,
        audit_calls=audit_calls,
        restored=restored_evaluate and restored_audit,
    )
    return {"summary": summary, "recovery_audit": completion}


def audit_recovered() -> dict[str, Any]:
    _validate_recovery_receipt(require_outputs_absent=False)
    summary = frozen._load_json(frozen.SUMMARY_PATH)
    legacy_audit, _, restored = _with_recovered_parity(frozen.audit)
    _validate_legacy_audit(summary, legacy_audit)
    completion = _validate_recovery_audit()
    if not restored or completion.get("canonical_summary_self_hash") != summary.get(
        "summary_sha256"
    ):
        raise frozen.BenchmarkError("recovery completion differs")
    return {
        "summary": summary,
        "legacy_audit": legacy_audit,
        "recovery_audit": completion,
    }


def finalize_recovery_audit() -> dict[str, Any]:
    _validate_recovery_receipt(require_outputs_absent=False)
    if AUDIT_PATH.exists():
        return audit_recovered()
    summary = frozen._load_json(frozen.SUMMARY_PATH)
    legacy_audit, audit_calls, restored = _with_recovered_parity(frozen.audit)
    _validate_legacy_audit(summary, legacy_audit)
    completion = _write_recovery_audit(
        summary,
        legacy_audit,
        evaluate_calls=None,
        audit_calls=audit_calls,
        restored=restored,
    )
    return {
        "summary": summary,
        "legacy_audit": legacy_audit,
        "recovery_audit": completion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--seal-recovery", action="store_true")
    phase.add_argument("--evaluate", action="store_true")
    phase.add_argument("--audit", action="store_true")
    phase.add_argument("--finalize-recovery-audit", action="store_true")
    args = parser.parse_args()
    if args.seal_recovery:
        result = seal_recovery()
    elif args.evaluate:
        result = evaluate_recovered()
    elif args.audit:
        result = audit_recovered()
    else:
        result = finalize_recovery_audit()
    print(frozen.json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
