import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from denoiseapt import automatic_runtime as automatic_runtime_module
from denoiseapt.api import DemoService
from denoiseapt.automatic_runtime import AutomaticPreservationRuntime


ROOT = Path(__file__).resolve().parents[1]
GUIDED_CASE_PATH = ROOT / "data" / "prepared" / "tsb_ad_ucr_medical_guided.npz"
guided_case_required = pytest.mark.skipif(
    not GUIDED_CASE_PATH.is_file(),
    reason="optional download-prepared guided benchmark case is not installed",
)


def _case(case_id: str):
    with np.load(ROOT / "data" / "prepared" / f"{case_id}.npz", allow_pickle=False) as item:
        signal = np.asarray(item["signal"], dtype=np.float32)
        metadata = json.loads(str(item["metadata_json"].item()))
    return signal, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_release_root(path: Path) -> Path:
    """Create the redistributable shape: core artifacts plus synthetic data."""

    (path / "config").mkdir(parents=True)
    (path / "data" / "prepared").mkdir(parents=True)
    shutil.copy2(ROOT / "config" / "demo.json", path / "config" / "demo.json")
    shutil.copy2(
        ROOT / "data" / "prepared" / "synthetic_guided_case.npz",
        path / "data" / "prepared" / "synthetic_guided_case.npz",
    )
    shutil.copytree(
        ROOT / "checkpoints" / "automatic_preservation",
        path / "checkpoints" / "automatic_preservation",
    )
    return path


@pytest.fixture(scope="module")
def runtime():
    return AutomaticPreservationRuntime(ROOT)


def test_runtime_loads_validation_selected_seed17_artifacts(runtime):
    assert runtime.selected_model == "denoiseapt_lambda_0p25"
    assert runtime.manifest["seed"] == 17
    assert runtime.manifest["controller"]["development_only"] is True
    assert runtime.manifest["controller"]["confirmation_results_packaged"] is False
    assert set(runtime.controller_config.required_witness_ids) == {
        "A_causal_mlp",
        "B_causal_conv",
    }
    prepared = runtime.manifest["prepared_case"]
    assert prepared["case_id"] == "tsb_ad_ucr_medical_guided"
    assert runtime.manifest["prepared_case_bundled"] is False


@guided_case_required
def test_optional_guided_case_matches_frozen_manifest(runtime):
    prepared = runtime.manifest["prepared_case"]
    assert _sha256(ROOT / prepared["path"]) == prepared["sha256"]
    assert runtime.guided_certificate_ready is True


@guided_case_required
def test_guided_case_is_deterministic_and_witness_bound(runtime):
    signal, metadata = _case("tsb_ad_ucr_medical_guided")
    first = runtime.run(
        signal,
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
    )
    second = runtime.run(
        signal,
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
    )
    assert first.eligibility.eligible is True
    assert first.eligibility.mode == "witness_certificate"
    assert first.projection is not None
    assert first.projection.certificate.status == "passed"
    assert np.array_equal(first.soft_candidate, second.soft_candidate)
    assert np.array_equal(first.ordinary_candidate, second.ordinary_candidate)
    assert np.array_equal(first.automatic, second.automatic)
    assert first.projection.audit.decision_hash == second.projection.audit.decision_hash


@guided_case_required
def test_hybrid_independent_score_mismatch_falls_back_exactly(runtime, monkeypatch):
    signal, metadata = _case("tsb_ad_ucr_medical_guided")
    inference = runtime.run(
        signal,
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
    )
    monkeypatch.setattr(
        automatic_runtime_module,
        "_verification_scores_equal",
        lambda *_: False,
    )
    hybrid = runtime.run_hybrid(inference)
    assert hybrid.auto_committed is False
    assert hybrid.fallback_reason == "independent_postcheck_score_mismatch"
    assert np.array_equal(hybrid.output, inference.observation)
    assert hybrid.independent_verification.passed is True


def test_synthetic_and_upload_like_inputs_are_review_only(runtime):
    signal, metadata = _case("synthetic_guided_case")
    packaged = runtime.run(
        signal,
        case_id="synthetic_guided_case",
        metadata=metadata,
        is_upload=False,
    )
    uploaded = runtime.run(
        signal,
        case_id="upload",
        metadata={"domain": "Uploaded"},
        is_upload=True,
    )
    for result in (packaged, uploaded):
        payload = result.automatic_payload()
        assert result.eligibility.eligible is False
        assert result.projection is None
        assert payload["mode"] == "review_only"
        assert payload["auto_committed"] is False
        assert payload["certificate"]["status"] == "unverified"
        assert np.array_equal(result.automatic, result.soft_candidate)


@guided_case_required
def test_guided_case_with_changed_window_contract_is_review_only(runtime):
    _, metadata = _case("tsb_ad_ucr_medical_guided")
    eligibility = runtime.assess_eligibility(
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
        window_length=256,
    )
    assert eligibility.eligible is False
    assert eligibility.mode == "review_only"
    assert "512-sample" in eligibility.reason


@guided_case_required
def test_guided_case_hash_failure_downgrades_to_review_only(runtime, monkeypatch):
    _, metadata = _case("tsb_ad_ucr_medical_guided")
    prepared = (ROOT / runtime.manifest["prepared_case"]["path"]).resolve()
    original = automatic_runtime_module.file_sha256

    def mismatched(path: Path) -> str:
        if path.resolve() == prepared:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(automatic_runtime_module, "file_sha256", mismatched)
    eligibility = runtime.assess_eligibility(
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
        window_length=512,
    )
    assert eligibility.eligible is False
    assert eligibility.mode == "review_only"
    assert "frozen hash check" in eligibility.reason


@guided_case_required
def test_session_restores_immutable_automatic_baseline(runtime):
    signal, metadata = _case("tsb_ad_ucr_medical_guided")
    result = runtime.run(
        signal,
        case_id="tsb_ad_ucr_medical_guided",
        metadata=metadata,
        is_upload=False,
    )
    session = result.new_session()
    baseline = session.current.copy()
    session.apply("protect", 220, 280)
    session.apply("restore_automatic")
    assert np.array_equal(session.current, baseline)
    assert session.to_dict()["certificate"]["status"] == "passed"


@guided_case_required
def test_api_defaults_approved_to_automatic_and_returns_boolean_certificate():
    service = DemoService(ROOT)
    response = service.analyze(
        {
            "case_id": "tsb_ad_ucr_medical_guided",
            "corruption": {"family": "mixed", "severity": 0.32, "seed": 17},
            "window": {"start": 0, "length": 512},
        }
    )
    assert response["series"]["approved"] == response["series"]["automatic"]
    assert response["automatic_control"]["mode"] == "witness_certificate"
    assert isinstance(response["automatic_control"]["auto_committed"], bool)
    assert isinstance(response["automatic_control"]["certificate"]["passed"], bool)
    assert response["hybrid_control"]["mode"] == "witness_certificate"
    assert response["hybrid_control"]["auto_committed"] is True
    assert response["hybrid_control"]["certificate"]["passed"] is True
    assert response["hybrid_control"]["independent_recheck_exact_match"] is True
    assert response["series"]["approved"] == response["series"]["automatic"]


@guided_case_required
def test_api_reports_human_override_then_restored_automatic_decision():
    service = DemoService(ROOT)
    analysis = service.analyze(
        {
            "case_id": "tsb_ad_ucr_medical_guided",
            "corruption": {"family": "impulse", "severity": 0.35, "seed": 17},
            "window": {"start": 0, "length": 512},
        }
    )
    baseline_decision = analysis["automatic_control"]["decision"]
    override = service.intervene(
        {
            "session_id": analysis["session_id"],
            "action": "protect",
            "start": 251,
            "end": 263,
            "expected_revision": analysis["revision"],
        }
    )
    assert override["automatic_control"]["decision"] == "human_override"
    assert override["automatic_control"]["auto_committed"] is False
    assert override["automatic_control"]["current_is_automatic"] is False
    assert override["automatic_control"]["human_intervention"]["action"] == "protect"

    restored = service.intervene(
        {
            "session_id": analysis["session_id"],
            "action": "restore_automatic",
            "expected_revision": override["revision"],
        }
    )
    assert restored["automatic_control"]["decision"] == baseline_decision
    assert restored["automatic_control"]["auto_committed"] is True
    assert restored["automatic_control"]["current_is_automatic"] is True


def test_api_upload_is_unverified_review_only_without_threshold_metrics():
    service = DemoService(ROOT)
    values = np.sin(np.linspace(0.0, 20.0, 257)).tolist()
    response = service.analyze(
        {
            "upload": {"name": "unseen.csv", "values": values},
            "corruption": {"family": "none", "severity": 0.0, "seed": 17},
            "window": {"start": 0, "length": 257},
        }
    )
    control = response["automatic_control"]
    assert control["mode"] == "review_only"
    assert control["certification_eligible"] is False
    assert control["auto_committed"] is False
    assert control["certificate"]["status"] == "unverified"
    assert response["series"]["approved"] == response["series"]["automatic"]
    assert len(response["series"]["hybrid"]) == 257
    assert response["hybrid_control"]["mode"] == "review_only"
    assert response["hybrid_control"]["auto_committed"] is False
    assert response["hybrid_control"]["certificate"]["status"] == "unverified"
    assert response["hybrid_control"]["denoiseapt_repair_source_kind"] == (
        "denoiseapt_soft_review_source"
    )
    assert response["hybrid_control"]["hybrid_latency_ms"] >= response[
        "hybrid_control"
    ]["routing_latency_ms"]
    assert response["meta"]["hybrid_algorithm_version"] == (
        "evidence-gated-classical-dapt-v1"
    )
    assert len(response["series"]["automatic"]) == 257
    assert "event_recall" not in response["metrics"]["automatic"]
    assert "score_threshold" not in response["metrics"]["automatic"]


def test_clean_release_without_guided_case_runs_synthetic_review_only(tmp_path):
    release_root = _clean_release_root(tmp_path / "release")
    assert not (
        release_root / "data" / "prepared" / "tsb_ad_ucr_medical_guided.npz"
    ).exists()

    service = DemoService(release_root)
    health = service.health()
    assert health["automatic_runtime_ready"] is True
    assert health["automatic_runtime_error"] is None
    assert health["guided_case_ready"] is False
    assert health["guided_certificate_ready"] is False
    assert "not installed" in health["guided_certificate_reason"]

    cases = {item["id"]: item for item in service.list_cases()["cases"]}
    assert set(cases) == {"synthetic_guided_case"}
    assert cases["synthetic_guided_case"]["automatic_certificate_available"] is False
    response = service.analyze(
        {
            "case_id": "synthetic_guided_case",
            "corruption": {"family": "mixed", "severity": 0.32, "seed": 17},
            "window": {"start": 0, "length": 512},
        }
    )
    control = response["automatic_control"]
    assert response["meta"]["review_only"] is True
    assert control["mode"] == "review_only"
    assert control["certification_eligible"] is False
    assert control["auto_committed"] is False
    assert control["certificate"]["status"] == "unverified"
    assert response["series"]["automatic"] == response["series"]["soft_candidate"]
    assert len(response["series"]["hybrid"]) == 512
    assert response["hybrid_control"]["mode"] == "review_only"
    assert response["hybrid_control"]["certification_eligible"] is False


def test_present_but_hash_mismatched_guided_case_stays_fail_closed(tmp_path):
    release_root = _clean_release_root(tmp_path / "release")
    guided_path = (
        release_root / "data" / "prepared" / "tsb_ad_ucr_medical_guided.npz"
    )
    guided_path.write_bytes(b"present but not the allowlisted prepared case")

    runtime = AutomaticPreservationRuntime(release_root)
    assert runtime.ready is True
    assert runtime.guided_certificate_ready is False
    assert "frozen hash" in runtime.guided_certificate_status["reason"]
    eligibility = runtime.assess_eligibility(
        case_id="tsb_ad_ucr_medical_guided",
        metadata={
            "source_file": "TSB-AD-U/442_UCR_id_140_Medical_tr_1875_1st_4187.csv"
        },
        is_upload=False,
        window_length=512,
    )
    assert eligibility.eligible is False
    assert eligibility.mode == "review_only"
    assert "frozen hash check" in eligibility.reason

    service = DemoService(release_root)
    health = service.health()
    assert health["automatic_runtime_ready"] is True
    assert health["guided_case_ready"] is True
    assert health["guided_certificate_ready"] is False
    catalog = service.list_cases()
    assert [case["id"] for case in catalog["cases"]] == ["synthetic_guided_case"]
    assert catalog["warnings"][0]["case_id"] == "tsb_ad_ucr_medical_guided"
