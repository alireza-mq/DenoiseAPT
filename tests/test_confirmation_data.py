from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from denoiseapt.confirmation_data import (
    build_confirmation_artifacts,
    verify_confirmation_lock,
)
from denoiseapt.experiment_data import normal_prefix_sha256
from denoiseapt.models import TemporalPatchDiscriminator, TemporalUNetGenerator


def _csv(values: np.ndarray, event: tuple[int, int]) -> bytes:
    labels = np.zeros(len(values), dtype=np.uint8)
    labels[event[0] : event[1]] = 1
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["Data", "Label"])
    writer.writerows(zip(values, labels))
    return stream.getvalue().encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    experiments = tmp_path / "experiments"
    data = tmp_path / "data" / "experiment"
    experiments.mkdir(parents=True)
    data.mkdir(parents=True)
    names = {
        "001_SMD_id_1_Facility_tr_64_1st_90.csv": np.linspace(0.0, 1.0, 128),
        "002_SMD_id_2_Facility_tr_64_1st_90.csv": np.linspace(1.0, 2.0, 128),
        "003_SMD_id_3_Facility_tr_64_1st_90.csv": np.linspace(2.0, 3.0, 128),
        "004_SMD_id_4_Facility_tr_64_1st_90.csv": np.linspace(3.0, 4.0, 128),
        "010_Exathlon_id_1_Facility_tr_64_1st_90.csv": np.sin(np.arange(128) / 7),
        "011_Exathlon_id_2_Facility_tr_64_1st_90.csv": np.sin(np.arange(128) / 9),
        "012_Exathlon_id_3_Facility_tr_64_1st_90.csv": np.sin(np.arange(128) / 11),
        # Exact-prefix twin of eval file 010: the complete group must be excluded.
        "013_Exathlon_id_4_Facility_tr_64_1st_92.csv": np.r_[
            np.sin(np.arange(64) / 7), np.cos(np.arange(64) / 5)
        ],
    }
    eval_names = sorted(name for name in names if not name.startswith(("004_", "013_")))
    tuning_names = [
        "004_SMD_id_4_Facility_tr_64_1st_90.csv",
        "013_Exathlon_id_4_Facility_tr_64_1st_92.csv",
    ]
    eval_path = experiments / "eval.csv"
    tuning_path = experiments / "tuning.csv"
    eval_path.write_text("file_name\n" + "\n".join(eval_names) + "\n", "utf-8")
    tuning_path.write_text("file_name\n" + "\n".join(tuning_names) + "\n", "utf-8")

    archive = tmp_path / "TSB-AD-U.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, values in names.items():
            event = (92, 100) if name.startswith("013_") else (90, 98)
            bundle.writestr(f"TSB-AD-U/{name}", _csv(values, event))

    prior_path = data / "protocol_v1_manifest.json"
    unrelated = normal_prefix_sha256(np.linspace(10.0, 11.0, 128), 64)
    prior_path.write_text(
        json.dumps(
            {
                "protocol_id": "spent-v1",
                "sources": [{"normal_prefix_sha256": unrelated}],
            }
        ),
        "utf-8",
    )
    config = {
        "protocol_id": "confirmation-fixture",
        "schema_version": 1,
        "sealed_utc_date": "2026-08-13",
        "seed": 9,
        "window_length": 32,
        "max_event_duration": 24,
        "minimum_normal_samples_in_event_window": 8,
        "calibration_groups_per_domain": 1,
        "minimum_confirmation_groups_per_domain": 1,
        "archive": {
            "url": "https://example.test/archive.zip",
            "bytes": archive.stat().st_size,
            "sha256": _sha256(archive),
        },
        "official_lists": {
            "repository_commit": "fixture",
            "evaluation": "eval.csv",
            "tuning": "tuning.csv",
            "evaluation_url": "https://example.test/eval.csv",
            "tuning_url": "https://example.test/tuning.csv",
            "evaluation_sha256": _sha256(eval_path),
            "tuning_sha256": _sha256(tuning_path),
            "evaluation_entries": len(eval_names),
            "tuning_entries": len(tuning_names),
        },
        "prior_spent_protocol": {
            "manifest": "../data/experiment/protocol_v1_manifest.json",
            "manifest_sha256": _sha256(prior_path),
        },
        "domains": [
            {
                "key": "smd",
                "source_collection": "SMD",
                "domain": "Facility",
                "confirmation_role": "primary",
            },
            {
                "key": "exathlon",
                "source_collection": "Exathlon",
                "domain": "Facility",
                "confirmation_role": "secondary",
            },
        ],
        "forbidden_spent_domains": [
            {"source_collection": "UCR", "domain": "Medical"},
            {"source_collection": "SMAP", "domain": "*"},
        ],
        "source_overlap_policy": {"fixture": "whole-group-disjoint"},
        "construction_only_notice": {"models": "none"},
        "provenance_and_licenses": {"fixture": True},
    }
    config_path = experiments / "protocol.json"
    config_path.write_text(json.dumps(config), "utf-8")
    return archive, config_path


def _build_at(tmp_path: Path, suffix: str = "") -> tuple[dict, Path, Path, Path]:
    archive, config = _write_fixture(tmp_path)
    output = tmp_path / f"confirmation{suffix}.npz"
    manifest = tmp_path / f"manifest{suffix}.json"
    lock = tmp_path / f"LOCK{suffix}.json"
    result = build_confirmation_artifacts(
        archive_path=archive,
        protocol_config=config,
        output_npz=output,
        output_manifest=manifest,
        output_lock=lock,
    )
    return result, output, manifest, lock


def test_confirmation_builder_is_whole_group_disjoint_and_prefix_only(
    tmp_path: Path,
) -> None:
    manifest, output, _, _ = _build_at(tmp_path)
    windows = manifest["windows"]
    calibration = {
        item["source_group_id"] for item in windows if item["partition"] == "calibration"
    }
    confirmation = {
        item["source_group_id"] for item in windows if item["partition"] == "confirmation"
    }
    assert calibration.isdisjoint(confirmation)
    assert all(
        item["kind"] == "normal" and item["positive_samples"] == 0
        for item in windows
        if item["partition"] == "calibration"
    )
    excluded_members = {
        member
        for item in manifest["exclusions"]
        if item["scope"] == "source_group"
        for member in item["members"]
    }
    assert "010_Exathlon_id_1_Facility_tr_64_1st_90.csv" in excluded_members
    assert "013_Exathlon_id_4_Facility_tr_64_1st_92.csv" in excluded_members
    with np.load(output, allow_pickle=False) as bundle:
        for domain in ("smd", "exathlon"):
            calibration_labels = bundle[f"{domain}_calibration_labels"]
            assert calibration_labels.shape[1] == 32
            assert not np.any(calibration_labels)
            assert set(bundle[f"{domain}_calibration_source_group_id"]).isdisjoint(
                set(bundle[f"{domain}_confirmation_source_group_id"])
            )
            assert set(bundle[f"{domain}_confirmation_domain_key"]) == {domain}
            assert all(
                value.endswith("/Facility")
                for value in bundle[f"{domain}_confirmation_source_domain"]
            )


def test_confirmation_npz_is_byte_deterministic(tmp_path: Path) -> None:
    _, first, _, _ = _build_at(tmp_path, "-a")
    # Reuse the same immutable inputs but write a separately named sealed copy.
    config = tmp_path / "experiments" / "protocol.json"
    archive = tmp_path / "TSB-AD-U.zip"
    second = tmp_path / "confirmation-b.npz"
    build_confirmation_artifacts(
        archive_path=archive,
        protocol_config=config,
        output_npz=second,
        output_manifest=tmp_path / "manifest-b.json",
        output_lock=tmp_path / "LOCK-b.json",
    )
    assert first.read_bytes() == second.read_bytes()


def test_sealed_confirmation_rejects_artifact_tampering(tmp_path: Path) -> None:
    _, output, manifest, lock = _build_at(tmp_path)
    output.write_bytes(output.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact_npz SHA-256 mismatch"):
        verify_confirmation_lock(
            archive_path=tmp_path / "TSB-AD-U.zip",
            protocol_config=tmp_path / "experiments" / "protocol.json",
            output_npz=output,
            output_manifest=manifest,
            output_lock=lock,
        )


def test_2048_window_is_architecture_compatible_without_loading_checkpoint() -> None:
    generator = TemporalUNetGenerator().eval()
    discriminator = TemporalPatchDiscriminator().eval()
    observation = torch.zeros((1, 1, 2048), dtype=torch.float32)
    with torch.no_grad():
        candidate = generator(observation)
        patches = discriminator(observation, candidate)
    assert candidate.shape == observation.shape
    assert patches.ndim == 3 and patches.shape[0:2] == (1, 1)
    assert patches.shape[-1] > 1
