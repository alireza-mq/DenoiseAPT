from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from denoiseapt.experiment_data import (
    build_protocol_artifacts,
    normal_prefix_sha256,
    parse_tsb_ad_filename,
)


def _csv(values: np.ndarray, event: tuple[int, int], *, prefix_positive: bool = False) -> bytes:
    labels = np.zeros(len(values), dtype=np.uint8)
    labels[event[0] : event[1]] = 1
    if prefix_positive:
        labels[10] = 1
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["Data", "Label"])
    writer.writerows(zip(values, labels))
    return stream.getvalue().encode("utf-8")


def _write_fixture(tmp_path: Path, *, prefix_positive: bool = False) -> tuple[Path, Path]:
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    evaluation_names = [
        "001_UCR_id_1_Medical_tr_600_1st_700.csv",
        "010_SMAP_id_1_Sensor_tr_600_1st_700.csv",
    ]
    tuning_names = ["003_UCR_id_3_Medical_tr_600_1st_700.csv"]
    (experiments / "eval.csv").write_text(
        "file_name\n" + "\n".join(evaluation_names) + "\n", encoding="utf-8"
    )
    (experiments / "tuning.csv").write_text(
        "file_name\n" + "\n".join(tuning_names) + "\n", encoding="utf-8"
    )
    config = {
        "protocol_id": "test-protocol",
        "schema_version": 1,
        "seed": 41,
        "window_length": 512,
        "max_event_duration": 384,
        "validation_fraction_of_development_groups": 0.2,
        "primary": {"source_collection": "UCR", "domain": "Medical"},
        "transfer": {"source_collection": "SMAP", "official_split": "evaluation"},
        "official_lists": {
            "evaluation": "eval.csv",
            "tuning": "tuning.csv",
            "evaluation_url": "https://example.test/eval.csv",
            "tuning_url": "https://example.test/tuning.csv",
        },
        "split_policy": {"test": "fixture"},
    }
    config_path = experiments / "protocol.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    common = np.linspace(-1.0, 1.0, 600)
    members = {
        "TSB-AD-U/001_UCR_id_1_Medical_tr_600_1st_700.csv": np.r_[
            common, np.linspace(2.0, 4.0, 600)
        ],
        # Different anomaly suffix, exact same prefix as official evaluation 001.
        "TSB-AD-U/002_UCR_id_2_Medical_tr_600_1st_720.csv": np.r_[
            common, np.linspace(-4.0, -2.0, 600)
        ],
        "TSB-AD-U/003_UCR_id_3_Medical_tr_600_1st_700.csv": np.linspace(2.0, 5.0, 1200),
        "TSB-AD-U/004_UCR_id_4_Medical_tr_600_1st_700.csv": np.sin(
            np.arange(1200) / 31.0
        ),
        "TSB-AD-U/005_UCR_id_5_Medical_tr_600_1st_700.csv": np.cos(
            np.arange(1200) / 29.0
        ),
        "TSB-AD-U/010_SMAP_id_1_Sensor_tr_600_1st_700.csv": np.sin(
            np.arange(1200) / 17.0
        ),
    }
    archive = tmp_path / "TSB-AD-U.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index, (member, values) in enumerate(members.items()):
            bundle.writestr(
                member,
                _csv(
                    values,
                    (700, 720),
                    prefix_positive=prefix_positive and index == 0,
                ),
            )
    return archive, config_path


def test_filename_parser_and_prefix_hash_are_stable() -> None:
    parsed = parse_tsb_ad_filename("305_UCR_id_3_Medical_tr_3000_1st_5948.csv")
    assert parsed.source_collection == "UCR"
    assert parsed.domain == "Medical"
    assert parsed.training_end == 3000
    values = np.arange(20, dtype=np.float64)
    assert normal_prefix_sha256(values, 10) == normal_prefix_sha256(values.copy(), 10)
    assert normal_prefix_sha256(values, 10) != normal_prefix_sha256(values, 11)


def test_protocol_keeps_duplicate_prefix_group_in_official_test(tmp_path: Path) -> None:
    archive, config = _write_fixture(tmp_path)
    output = tmp_path / "protocol.npz"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_protocol_artifacts(
        archive_path=archive,
        protocol_config=config,
        output_npz=output,
        output_manifest=manifest_path,
    )
    sources = {item["file_name"]: item for item in manifest["sources"]}
    assert sources["001_UCR_id_1_Medical_tr_600_1st_700.csv"]["assigned_split"] == "test"
    assert sources["002_UCR_id_2_Medical_tr_600_1st_720.csv"]["assigned_split"] == "test"
    assert (
        sources["001_UCR_id_1_Medical_tr_600_1st_700.csv"]["source_group_id"]
        == sources["002_UCR_id_2_Medical_tr_600_1st_720.csv"]["source_group_id"]
    )
    assert sources["003_UCR_id_3_Medical_tr_600_1st_700.csv"]["assigned_split"] == "validation"
    assert {
        sources["004_UCR_id_4_Medical_tr_600_1st_700.csv"]["assigned_split"],
        sources["005_UCR_id_5_Medical_tr_600_1st_700.csv"]["assigned_split"],
    } == {"train"}
    assert not (set(manifest["split_groups"]["train"]) & set(manifest["split_groups"]["test"]))

    with np.load(output, allow_pickle=False) as bundle:
        expected = {
            "train_clean",
            "train_labels",
            "train_series_id",
            "validation_clean",
            "validation_labels",
            "validation_series_id",
            "test_clean",
            "test_labels",
            "test_series_id",
            "transfer_clean",
            "transfer_labels",
            "transfer_series_id",
            "metadata_json",
        }
        assert expected <= set(bundle.files)
        for split in ("train", "validation", "test", "transfer"):
            assert bundle[f"{split}_clean"].shape[1] == 512
            assert bundle[f"{split}_labels"].shape == bundle[f"{split}_clean"].shape
        assert json.loads(str(bundle["metadata_json"].item()))["normalised"] is False


def test_protocol_is_deterministic_at_assignment_and_array_level(tmp_path: Path) -> None:
    archive, config = _write_fixture(tmp_path)
    manifests = []
    arrays = []
    for index in range(2):
        output = tmp_path / f"protocol-{index}.npz"
        manifests.append(
            build_protocol_artifacts(
                archive_path=archive,
                protocol_config=config,
                output_npz=output,
                output_manifest=tmp_path / f"manifest-{index}.json",
            )
        )
        with np.load(output, allow_pickle=False) as bundle:
            arrays.append({key: bundle[key].copy() for key in bundle.files})
    assert manifests[0]["split_groups"] == manifests[1]["split_groups"]
    assert manifests[0]["windows"] == manifests[1]["windows"]
    for key in arrays[0]:
        np.testing.assert_array_equal(arrays[0][key], arrays[1][key])


def test_positive_label_in_declared_normal_prefix_is_rejected(tmp_path: Path) -> None:
    archive, config = _write_fixture(tmp_path, prefix_positive=True)
    with pytest.raises(ValueError, match="normal training prefix"):
        build_protocol_artifacts(
            archive_path=archive,
            protocol_config=config,
            output_npz=tmp_path / "protocol.npz",
            output_manifest=tmp_path / "manifest.json",
        )


def test_tampered_official_split_list_is_rejected_when_hash_is_pinned(
    tmp_path: Path,
) -> None:
    archive, config_path = _write_fixture(tmp_path)
    config = json.loads(config_path.read_text("utf-8"))
    config["official_lists"]["evaluation_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch for official evaluation list"):
        build_protocol_artifacts(
            archive_path=archive,
            protocol_config=config_path,
            output_npz=tmp_path / "protocol.npz",
            output_manifest=tmp_path / "manifest.json",
        )
