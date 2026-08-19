from __future__ import annotations

import hashlib
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

from denoiseapt.datasets import (
    TimeSeriesCase,
    apply_corruption,
    builtin_demo_case,
    download_tsb_ad_u,
    extract_window,
    iter_training_cases,
    load_csv,
    prepare_case,
    secure_extract_zip,
    write_case_csv,
    write_case_npz,
)


def test_builtin_case_is_deterministic_and_labelled() -> None:
    first = builtin_demo_case()
    second = builtin_demo_case()
    np.testing.assert_array_equal(first.signal, second.signal)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert len(first.signal) == 512
    assert first.labels.any()
    assert first.metadata["synthetic"] is True


def test_csv_round_trip_and_tsb_style_header(tmp_path: Path) -> None:
    original = builtin_demo_case(128)
    portable = write_case_csv(original, tmp_path / "portable.csv")
    loaded = load_csv(portable)
    np.testing.assert_allclose(loaded.signal, original.signal)
    np.testing.assert_array_equal(loaded.labels, original.labels)

    tsb = tmp_path / "tsb.csv"
    tsb.write_text("Data,Label\n1.25,0\n2.5,1\n3.75,0\n", encoding="utf-8")
    loaded_tsb = load_csv(tsb)
    np.testing.assert_allclose(loaded_tsb.signal, [1.25, 2.5, 3.75])
    np.testing.assert_array_equal(loaded_tsb.labels, [False, True, False])


def test_npz_schema_is_pickle_free(tmp_path: Path) -> None:
    case = builtin_demo_case(128)
    path = write_case_npz(case, tmp_path / "case.npz", {"domain": "Synthetic"})
    with np.load(path, allow_pickle=False) as item:
        assert set(item.files) == {"signal", "labels", "metadata_json"}
        np.testing.assert_allclose(item["signal"], case.signal)
        assert '"domain": "Synthetic"' in str(item["metadata_json"].item())


def test_corruptions_are_seeded_and_non_mutating() -> None:
    signal = builtin_demo_case().signal
    original = signal.copy()
    for kind in ["gaussian", "impulse", "drift", "mixed"]:
        one = apply_corruption(signal, kind, severity=0.4, seed=19)
        two = apply_corruption(signal, kind, severity=0.4, seed=19)
        np.testing.assert_array_equal(one, two)
        assert not np.array_equal(one, signal)
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_allclose(apply_corruption(signal, "none", 0.9), signal, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="severity"):
        apply_corruption(signal, "mixed", 1.1)


def test_automatic_window_centres_event_and_records_source_indices() -> None:
    signal = np.sin(np.arange(1000) / 20)
    labels = np.zeros(1000, dtype=bool)
    labels[700:712] = True
    case = TimeSeriesCase("long", np.arange(1000), signal, labels)
    window = extract_window(case, 200)
    assert len(window.signal) == 200
    assert window.metadata["source_window"] == [605, 805]
    assert window.metadata["source_index_start"] == 605
    assert window.metadata["source_index_stop"] == 805
    np.testing.assert_array_equal(np.flatnonzero(window.labels), np.arange(95, 107))


def test_prepare_case_returns_reference_and_observation() -> None:
    prepared = prepare_case(builtin_demo_case(), corruption="mixed", severity=0.2, seed=5)
    assert prepared.corruption == "mixed"
    assert prepared.seed == 5
    assert prepared.observation.shape == prepared.source.signal.shape
    assert not np.array_equal(prepared.observation, prepared.source.signal)
    payload = prepared.to_dict()
    assert len(payload["reference"]) == len(payload["observation"])


def test_secure_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("safe/data.csv", "Data,Label\n1,0\n2,1\n")
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        secure_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_secure_zip_extraction_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")
    with pytest.raises(ValueError, match="Symbolic|symbolic"):
        secure_extract_zip(archive, tmp_path / "output")


def test_cached_archive_subset_and_provenance(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / "TSB-AD-U.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("TSB-AD-U/001_NAB_case.csv", "Data,Label\n1,0\n2,1\n")
        bundle.writestr("TSB-AD-U/002_UCR_case.csv", "Data,Label\n3,0\n4,1\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = tmp_path / "dataset"
    provenance = download_tsb_ad_u(
        destination,
        cache_dir=cache,
        subset=["NAB"],
        expected_sha256=digest,
    )
    assert provenance["selected_member_count"] == 1
    assert (destination / "TSB-AD-U" / "001_NAB_case.csv").exists()
    assert not (destination / "TSB-AD-U" / "002_UCR_case.csv").exists()
    assert (destination / "PROVENANCE.json").exists()


def test_insecure_download_requires_nonempty_trusted_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty 64-character"):
        download_tsb_ad_u(
            tmp_path / "dataset",
            cache_dir=tmp_path / "cache",
            expected_sha256="",
            allow_insecure=True,
        )


def test_iter_training_cases_is_sorted_and_bounded(tmp_path: Path) -> None:
    for filename, offset in [("b.csv", 10), ("a.csv", 0)]:
        rows = "\n".join(f"{offset + index},0" for index in range(5))
        (tmp_path / filename).write_text(f"Data,Label\n{rows}\n", encoding="utf-8")
    cases = list(iter_training_cases(tmp_path, max_cases=1, min_length=5))
    assert [case.name for case in cases] == ["a"]
