import hashlib
import json
from pathlib import Path

import numpy as np

from denoiseapt.api import DemoService
from denoiseapt.benchmark_replay import load_benchmark_replay


ROOT = Path(__file__).resolve().parents[1]
CATS_PATH = ROOT / "data" / "prepared" / "tsb_ad_cats_heldout_replay.npz"
MITDB_PATH = (
    ROOT / "data" / "prepared" / "tsb_ad_mitdb_anomaly_preservation_replay.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "window": {"start": 0, "length": 512},
        "corruption": {"family": "gaussian", "severity": 0.75, "replicate": 0},
    }


def test_bundled_artifacts_match_manifests_and_scope():
    expected = {
        CATS_PATH: (True, 379, 411),
        MITDB_PATH: (False, 228, 237),
    }
    for path, (synthetic, start, end) in expected.items():
        manifest = json.loads(
            path.with_name(f"{path.stem}_manifest.json").read_text("utf-8")
        )
        assert _sha256(path) == manifest["artifact_sha256"]
        replay = load_benchmark_replay(path)
        assert replay.length == 512
        assert len(replay.condition_id) == 25
        assert int(replay.identity.sum()) == 1
        assert replay.metadata["held_out"] is True
        assert replay.metadata["synthetic"] is synthetic
        assert replay.metadata["expert_interval_start"] == start
        assert replay.metadata["expert_interval_end"] == end


def test_bundled_case_names_are_short_and_unambiguous():
    catalog = DemoService(ROOT).list_cases()
    names = [item["name"] for item in catalog["cases"] if item["benchmark_replay"]]
    assert names == ["CATS rich dynamics", "MITDB ECG"]
    assert all(
        "TSB-AD-U" not in name and "held-out replay" not in name
        for name in names
    )


def test_cats_main_workflow_metrics_and_reversible_edit():
    service = DemoService(ROOT)
    result = service.analyze(_request("tsb_ad_cats_heldout_replay"))
    assert result["automatic_control"]["certificate"]["passed"] is True
    np.testing.assert_allclose(
        [
            result["metrics"]["our_model"]["overall_os_nrmse"],
            result["metrics"]["our_model"]["anomaly_os_nrmse"],
        ],
        [0.11646172858176247, 0.1104215419030915],
        rtol=0.0,
        atol=1e-9,
    )
    adapted = service.intervene(
        {
            "session_id": result["session_id"],
            "action": "blend",
            "start": 379,
            "end": 411,
            "beta": 0.75,
            "expected_revision": 0,
        }
    )
    assert adapted["automatic_control"]["certificate"]["passed"] is True
    assert adapted["metrics"]["our_model"]["overall_os_nrmse"] < result["metrics"]["our_model"]["overall_os_nrmse"]


def test_mitdb_replay_preserves_both_configured_opportunities():
    service = DemoService(ROOT)
    result = service.analyze(_request("tsb_ad_mitdb_anomaly_preservation_replay"))
    np.testing.assert_allclose(
        [
            result["metrics"]["our_model"]["overall_os_nrmse"],
            result["metrics"]["our_model"]["anomaly_os_nrmse"],
        ],
        [0.12348061317688942, 0.09456453812041848],
        rtol=0.0,
        atol=1e-9,
    )
    assert result["automatic_control"]["certificate"]["passed"] is True
    session = service._sessions[result["session_id"]]
    retained = {name: 0 for name in result["series"]}
    event = slice(113, 400)
    for witness in session["witnesses"]:
        for name, values in result["series"].items():
            scores = witness.scorer(np.asarray(values, dtype=np.float32))
            retained[name] += int(float(np.max(scores[event])) >= witness.threshold)
    assert retained == {
        "reference": 2,
        "observed": 2,
        "median": 0,
        "wavelet": 1,
        "noisereduce": 2,
        "rins_t": 0,
        "our_model": 2,
    }
