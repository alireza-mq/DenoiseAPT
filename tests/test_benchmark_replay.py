import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from denoiseapt.benchmark_replay import BenchmarkReplaySession, load_benchmark_replay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_replay(tmp_path: Path) -> Path:
    path = tmp_path / "fixture_replay.npz"
    length = 512
    count = 25
    time = np.linspace(0.0, 8.0 * np.pi, length, dtype=np.float32)
    reference = np.sin(time).astype(np.float32)
    labels = np.zeros(length, dtype=np.int8)
    labels[160:208] = 1
    identity = np.zeros(count, dtype=np.int8)
    identity[-1] = 1
    families = np.array(
        ["gaussian"] * 6
        + ["impulse"] * 6
        + ["drift"] * 6
        + ["mixed"] * 6
        + ["none"]
    )
    severity = np.tile(np.repeat([0.25, 0.50, 0.75], 2), 4).astype(np.float32)
    severity = np.concatenate([severity, np.array([0.0], dtype=np.float32)])
    replicate = np.tile([0, 1], 12).astype(np.int32)
    replicate = np.concatenate([replicate, np.array([0], dtype=np.int32)])
    condition_id = np.array([f"condition-{index:02d}" for index in range(count)])
    condition_id[-1] = "identity"
    series = np.tile(reference, (count, 1)).astype(np.float32)
    metadata = {
        "schema_version": 1,
        "default_condition_id": "condition-04",
    }
    np.savez(
        path,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        reference=reference,
        labels=labels,
        condition_id=condition_id,
        family=families,
        severity=severity,
        replicate=replicate,
        identity=identity,
        center=np.zeros(count, dtype=np.float32),
        scale=np.ones(count, dtype=np.float32),
        observation=series,
        median_filter_w3=series,
        wavelet_shrinkage=series,
        noisereduce=series,
        rins_t=series,
        our_model=series,
    )
    manifest = {
        "schema_version": 1,
        "artifact": path.name,
        "artifact_sha256": _sha256(path),
    }
    path.with_name(f"{path.stem}_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def test_integrity_checked_replay_loads_without_the_excluded_public_npz(tmp_path):
    path = _write_replay(tmp_path)
    replay = load_benchmark_replay(path)
    assert replay.length == 512
    assert replay.default_index == 4
    assert replay.condition_index("gaussian", 0.75, 0) == 4
    assert int(replay.identity.sum()) == 1


def test_replay_integrity_fails_closed_after_artifact_tampering(tmp_path):
    path = _write_replay(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity manifest"):
        load_benchmark_replay(path)


def test_replay_session_accumulates_and_reverts_expert_edits():
    observation = np.arange(8, dtype=np.float32)
    automatic = np.full(8, -1.0, dtype=np.float32)
    session = BenchmarkReplaySession(observation, automatic)
    session.apply("blend", start=1, end=3, beta=1.0, expected_revision=0)
    session.apply("blend", start=5, end=7, beta=1.0, expected_revision=1)
    np.testing.assert_array_equal(
        session.current,
        np.array([-1, 1, 2, -1, -1, 5, 6, -1], dtype=np.float32),
    )
    np.testing.assert_array_equal(session.baseline, automatic)
    session.apply("revert", expected_revision=2)
    np.testing.assert_array_equal(
        session.current,
        np.array([-1, 1, 2, -1, -1, -1, -1, -1], dtype=np.float32),
    )
