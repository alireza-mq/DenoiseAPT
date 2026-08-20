from pathlib import Path

import numpy as np
import pytest

from denoiseapt.api import (
    ApiError,
    DemoService,
    _intervals,
    _load_prepared_case,
    _moving_average,
    _validated_signal,
)


ROOT = Path(__file__).resolve().parents[1]


def test_intervals_use_half_open_bounds():
    labels = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    assert _intervals(labels) == [
        {"start": 1, "end": 3, "label": "labeled anomaly"},
        {"start": 4, "end": 5, "label": "labeled anomaly"},
    ]


def test_moving_average_preserves_length():
    values = np.arange(20, dtype=float)
    assert len(_moving_average(values, 7)) == len(values)


def test_signal_validation_rejects_constant_and_nonfinite():
    with pytest.raises(ApiError):
        _validated_signal([1.0] * 100)
    with pytest.raises(ApiError):
        _validated_signal([0.0, float("nan"), 1.0])


def test_prepared_case_loader_rejects_mismatched_labels(tmp_path):
    path = tmp_path / "bad_case.npz"
    np.savez(
        path,
        signal=np.arange(64, dtype=np.float32),
        labels=np.zeros(63, dtype=np.int8),
    )
    with pytest.raises(ValueError, match="equal length"):
        _load_prepared_case(path)


def test_catalog_preserves_synthetic_and_benchmark_metadata():
    cases = {item["id"]: item for item in DemoService(ROOT).list_cases()["cases"]}
    synthetic = cases["synthetic_guided_case"]
    assert synthetic["synthetic"] is True
    assert synthetic["benchmark_case"] is False
    if "tsb_ad_ucr_medical_guided" in cases:
        benchmark = cases["tsb_ad_ucr_medical_guided"]
        assert benchmark["synthetic"] is False
        assert benchmark["benchmark_case"] is True


def test_bundled_heldout_replays_are_ready_and_ordered():
    service = DemoService(ROOT)
    health = service.health()
    assert health["heldout_replay_ready"] is True
    assert health["heldout_replay_count"] == 2
    catalog = service.list_cases()
    replay_ids = [
        case["id"] for case in catalog["cases"] if case["benchmark_replay"]
    ]
    assert replay_ids == [
        "tsb_ad_cats_heldout_replay",
        "tsb_ad_mitdb_anomaly_preservation_replay",
    ]
    assert [
        case["name"] for case in catalog["cases"] if case["benchmark_replay"]
    ] == ["CATS rich dynamics", "MITDB ECG"]
    assert catalog["default_case_id"] == "tsb_ad_cats_heldout_replay"
    assert catalog["warnings"] == []


def test_optional_msl_replay_is_manifested_but_not_redistributed():
    manifest = ROOT / "data" / "prepared" / "tsb_ad_msl_heldout_replay_manifest.json"
    artifact = ROOT / "data" / "prepared" / "tsb_ad_msl_heldout_replay.npz"
    assert manifest.exists()
    assert not artifact.exists()
    assert "intentionally omitted" in manifest.read_text("utf-8")


def test_csv_upload_remains_review_only_with_bundled_replays():
    values = np.sin(np.linspace(0.0, 12.0, 128)).tolist()
    response = DemoService(ROOT).analyze(
        {
            "upload": {"name": "unseen.csv", "values": values},
            "corruption": {"family": "none", "severity": 0.0, "seed": 17},
            "window": {"start": 0, "length": 128},
        }
    )
    assert response["meta"]["case_id"] == "upload"
    assert response["meta"]["review_only"] is True
    assert response["automatic_control"]["mode"] == "review_only"
    assert response["automatic_control"]["certification_eligible"] is False
    assert response["automatic_control"]["certificate"]["status"] == "unverified"
    assert response["hybrid_control"]["mode"] == "review_only"


def test_analyze_rejects_ambiguous_source_and_bad_timestamp_length():
    service = DemoService(ROOT)
    values = np.sin(np.linspace(0, 8, 64)).tolist()
    with pytest.raises(ApiError, match="either upload or case_id"):
        service.analyze(
            {"case_id": "synthetic_guided_case", "upload": {"values": values}}
        )
    with pytest.raises(ApiError, match="Timestamps and signal values"):
        service.analyze(
            {"upload": {"values": values, "timestamps": ["only-one"]}}
        )


@pytest.mark.parametrize("field", ["window", "corruption"])
def test_analyze_rejects_non_object_nested_requests(field):
    service = DemoService(ROOT)
    with pytest.raises(ApiError, match=rf"{field} must be a JSON object"):
        service.analyze(
            {
                "case_id": "synthetic_guided_case",
                field: ["not", "an", "object"],
            }
        )
