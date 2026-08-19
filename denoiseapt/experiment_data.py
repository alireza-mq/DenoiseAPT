"""Leakage-safe data manifests for the controlled DenoiseAPT experiment.

The TSB-AD UCR collection contains files with different anomaly locations but
identical normal prefixes.  A file-level random split can therefore put the same
base signal in both development and evaluation.  This module hashes the exact
numeric normal prefix declared by ``tr_`` in each TSB-AD filename and assigns
the complete hash group to one split before any windows are extracted.

The module writes raw, unnormalised reference windows.  Corruption and
normalisation belong to the frozen experiment runner so every method receives
the same observation and preprocessing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from numpy.typing import NDArray


_FILENAME = re.compile(
    r"^(?P<global_id>\d{3})_(?P<source>[^_]+)_id_(?P<source_id>\d+)_"
    r"(?P<domain>[^_]+)_tr_(?P<training_end>\d+)_1st_"
    r"(?P<first_anomaly>\d+)\.csv$"
)
_LABEL_NAMES = ("label", "labels", "anomaly", "is_anomaly", "ground_truth", "gt")
_SIGNAL_NAMES = ("value", "data", "signal", "measurement", "metric")
_PREFIX_HASH_DOMAIN = b"DenoiseAPT-normal-prefix-float64le-v1\0"


@dataclass(frozen=True)
class ParsedName:
    file_name: str
    global_id: int
    source_collection: str
    source_id: int
    domain: str
    training_end: int
    first_anomaly: int


@dataclass(frozen=True)
class SourceSeries:
    member: str
    parsed: ParsedName
    signal: NDArray[np.float64]
    labels: NDArray[np.uint8]
    file_sha256: str
    normal_prefix_sha256: str
    source_group_id: str
    official_split: str


@dataclass(frozen=True)
class WindowRecord:
    split: str
    window_id: str
    series_id: str
    source_group_id: str
    archive_member: str
    kind: str
    start: int
    end: int
    event_index: int | None
    signal: NDArray[np.float32]
    labels: NDArray[np.uint8]

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "window_id": self.window_id,
            "series_id": self.series_id,
            "source_group_id": self.source_group_id,
            "archive_member": self.archive_member,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "event_index": self.event_index,
            "positive_samples": int(self.labels.sum()),
        }


def parse_tsb_ad_filename(file_name: str) -> ParsedName:
    """Parse the metadata encoded by a canonical TSB-AD-U filename."""

    match = _FILENAME.fullmatch(Path(file_name).name)
    if match is None:
        raise ValueError(f"Not a canonical TSB-AD-U filename: {file_name!r}")
    fields = match.groupdict()
    return ParsedName(
        file_name=Path(file_name).name,
        global_id=int(fields["global_id"]),
        source_collection=fields["source"],
        source_id=int(fields["source_id"]),
        domain=fields["domain"],
        training_end=int(fields["training_end"]),
        first_anomaly=int(fields["first_anomaly"]),
    )


def read_official_file_list(path: str | Path) -> set[str]:
    """Read a one-column official TSB-AD file list."""

    source = Path(path)
    rows = [line.strip() for line in source.read_text("utf-8-sig").splitlines()]
    names = {Path(row).name for row in rows if row and row.casefold() != "file_name"}
    if not names:
        raise ValueError(f"Official file list is empty: {source}")
    for name in names:
        parse_tsb_ad_filename(name)
    return names


def normal_prefix_sha256(signal: Sequence[float] | np.ndarray, training_end: int) -> str:
    """Hash exact prefix samples using a platform-independent byte encoding."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not 1 <= int(training_end) <= len(values):
        raise ValueError("training_end must select a non-empty in-range prefix")
    canonical = np.ascontiguousarray(values[: int(training_end)], dtype="<f8")
    digest = hashlib.sha256()
    digest.update(_PREFIX_HASH_DOMAIN)
    digest.update(int(training_end).to_bytes(8, "little", signed=False))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def event_intervals(labels: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    binary = (np.asarray(labels).reshape(-1) > 0).astype(np.int8)
    changes = np.diff(np.r_[0, binary, 0])
    return list(
        zip(
            np.flatnonzero(changes == 1).astype(int).tolist(),
            np.flatnonzero(changes == -1).astype(int).tolist(),
        )
    )


def build_protocol_artifacts(
    *,
    archive_path: str | Path,
    protocol_config: str | Path | Mapping[str, Any],
    output_npz: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    """Create the fixed-length NPZ and full split/provenance manifest."""

    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_npz).expanduser().resolve()
    manifest_path = Path(output_manifest).expanduser().resolve()
    config, config_path = _load_config(protocol_config)
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    official = config["official_lists"]
    eval_path = (base_dir / official["evaluation"]).resolve()
    tuning_path = (base_dir / official["tuning"]).resolve()
    _verify_expected_sha256(
        eval_path, official.get("evaluation_sha256"), "official evaluation list"
    )
    _verify_expected_sha256(
        tuning_path, official.get("tuning_sha256"), "official tuning list"
    )
    eval_names = read_official_file_list(eval_path)
    tuning_names = read_official_file_list(tuning_path)
    for role, names in (("evaluation", eval_names), ("tuning", tuning_names)):
        expected_entries = official.get(f"{role}_entries")
        if expected_entries is not None and len(names) != int(expected_entries):
            raise ValueError(
                f"Official {role} list has {len(names)} entries; "
                f"expected {int(expected_entries)}"
            )
    overlap = eval_names & tuning_names
    if overlap:
        raise ValueError(f"Official evaluation and tuning lists overlap: {sorted(overlap)[:3]}")

    primary_spec = config["primary"]
    transfer_spec = config["transfer"]
    primary: list[SourceSeries] = []
    transfer: list[SourceSeries] = []
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not info.filename.casefold().endswith(".csv"):
                continue
            try:
                parsed = parse_tsb_ad_filename(Path(info.filename).name)
            except ValueError:
                continue
            is_primary = (
                parsed.source_collection == primary_spec["source_collection"]
                and parsed.domain == primary_spec["domain"]
            )
            is_transfer = (
                parsed.source_collection == transfer_spec["source_collection"]
                and parsed.file_name in eval_names
            )
            if not is_primary and not is_transfer:
                continue
            raw = bundle.read(info)
            signal, labels = _read_tsb_csv(raw, parsed.file_name)
            if parsed.training_end > len(signal):
                raise ValueError(
                    f"Filename training boundary exceeds series length for {parsed.file_name}"
                )
            if np.any(labels[: parsed.training_end]):
                raise ValueError(
                    f"Declared normal training prefix contains positive labels: {parsed.file_name}"
                )
            prefix_hash = normal_prefix_sha256(signal, parsed.training_end)
            official_split = (
                "evaluation"
                if parsed.file_name in eval_names
                else "tuning"
                if parsed.file_name in tuning_names
                else "unlisted"
            )
            record = SourceSeries(
                member=info.filename,
                parsed=parsed,
                signal=signal,
                labels=labels,
                file_sha256=hashlib.sha256(raw).hexdigest(),
                normal_prefix_sha256=prefix_hash,
                source_group_id=(
                    f"{parsed.source_collection.casefold()}-"
                    f"{parsed.domain.casefold()}-{prefix_hash}"
                ),
                official_split=official_split,
            )
            if is_primary:
                primary.append(record)
            if is_transfer:
                transfer.append(record)

    if not primary:
        raise ValueError("No primary series matched the protocol")
    if not transfer:
        raise ValueError("No official evaluation transfer series matched the protocol")

    assignments, reasons = _assign_primary_groups(primary, config)
    window_length = int(config["window_length"])
    max_event_duration = int(config["max_event_duration"])
    windows: list[WindowRecord] = []
    records_manifest: list[dict[str, Any]] = []

    for record in sorted(primary, key=lambda item: item.parsed.file_name):
        split = assignments[record.source_group_id]
        series_windows = _event_windows(record, split, window_length, max_event_duration)
        windows.extend(series_windows)
        records_manifest.append(
            _source_manifest(
                record,
                split,
                reasons[record.source_group_id],
                series_windows,
                max_event_duration,
            )
        )
    _append_one_normal_window_per_group(primary, assignments, windows, window_length)

    transfer_windows: list[WindowRecord] = []
    for record in sorted(transfer, key=lambda item: item.parsed.file_name):
        series_windows = _event_windows(
            record, "transfer", window_length, max_event_duration
        )
        transfer_windows.extend(series_windows)
        records_manifest.append(
            _source_manifest(
                record,
                "transfer",
                "official evaluation transfer",
                series_windows,
                max_event_duration,
            )
        )
    transfer_assignments = {record.source_group_id: "transfer" for record in transfer}
    _append_one_normal_window_per_group(
        transfer, transfer_assignments, transfer_windows, window_length
    )
    windows.extend(transfer_windows)

    for required in ("train", "validation", "test", "transfer"):
        if not any(window.split == required for window in windows):
            raise ValueError(f"Protocol produced no {required} windows")

    split_groups = {
        split: sorted(
            {window.source_group_id for window in windows if window.split == split}
        )
        for split in ("train", "validation", "test", "transfer")
    }
    _assert_group_disjoint(split_groups)

    manifest: dict[str, Any] = {
        "schema_version": int(config.get("schema_version", 1)),
        "protocol_id": config["protocol_id"],
        "seed": int(config["seed"]),
        "window_length": window_length,
        "max_event_duration": max_event_duration,
        "archive": {
            "file_name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
        },
        "official_lists": {
            "evaluation": {
                "file_name": eval_path.name,
                "sha256": _sha256_file(eval_path),
                "url": official.get("evaluation_url"),
                "entries": len(eval_names),
            },
            "tuning": {
                "file_name": tuning_path.name,
                "sha256": _sha256_file(tuning_path),
                "url": official.get("tuning_url"),
                "entries": len(tuning_names),
            },
        },
        "grouping": {
            "algorithm": "sha256-domain-separated-float64le-normal-prefix-v1",
            "prefix_boundary": "integer encoded by tr_ in the source filename",
            "group_disjointness_verified": True,
        },
        "split_policy": config["split_policy"],
        "summary": _summary(records_manifest, windows),
        "split_groups": split_groups,
        "sources": records_manifest,
        "windows": [window.manifest_dict() for window in windows],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    npz_metadata = {
        "schema_version": manifest["schema_version"],
        "protocol_id": manifest["protocol_id"],
        "window_length": window_length,
        "raw_reference_values": True,
        "normalised": False,
        "split_summary": manifest["summary"]["windows_by_split"],
    }
    arrays: dict[str, Any] = {"metadata_json": np.asarray(json.dumps(npz_metadata, sort_keys=True))}
    for split, prefix in (
        ("train", "train"),
        ("validation", "validation"),
        ("test", "test"),
        ("transfer", "transfer"),
    ):
        selected = [window for window in windows if window.split == split]
        arrays[f"{prefix}_clean"] = np.stack([window.signal for window in selected]).astype(
            np.float32
        )
        arrays[f"{prefix}_labels"] = np.stack([window.labels for window in selected]).astype(
            np.uint8
        )
        arrays[f"{prefix}_series_id"] = np.asarray(
            [window.series_id for window in selected]
        )
        arrays[f"{prefix}_source_group_id"] = np.asarray(
            [window.source_group_id for window in selected]
        )
        arrays[f"{prefix}_window_id"] = np.asarray(
            [window.window_id for window in selected]
        )
    np.savez_compressed(output, **arrays)
    manifest["artifact"] = {
        "file_name": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256_file(output),
        "arrays": sorted(arrays),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_config(
    value: str | Path | Mapping[str, Any]
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value).expanduser().resolve()
    return json.loads(path.read_text("utf-8")), path


def _read_tsb_csv(raw: bytes, file_name: str) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise ValueError(f"CSV has no header: {file_name}")
    canonical = {name.strip().casefold(): name for name in reader.fieldnames}
    label_name = next((canonical[name] for name in _LABEL_NAMES if name in canonical), None)
    if label_name is None:
        raise ValueError(f"CSV has no recognised label column: {file_name}")
    signal_name = next((canonical[name] for name in _SIGNAL_NAMES if name in canonical), None)
    if signal_name is None:
        signal_name = next(name for name in reader.fieldnames if name != label_name)
    signal: list[float] = []
    labels: list[int] = []
    for row_number, row in enumerate(reader, start=2):
        try:
            value = float((row.get(signal_name) or "").strip())
            label_value = float((row.get(label_name) or "0").strip())
        except ValueError as exc:
            raise ValueError(f"Malformed numeric value in {file_name}:{row_number}") from exc
        if not math.isfinite(value) or not math.isfinite(label_value):
            raise ValueError(f"Non-finite value in {file_name}:{row_number}")
        signal.append(value)
        labels.append(int(label_value != 0.0))
    if len(signal) < 2:
        raise ValueError(f"CSV has too few samples: {file_name}")
    return np.asarray(signal, dtype=np.float64), np.asarray(labels, dtype=np.uint8)


def _assign_primary_groups(
    records: Sequence[SourceSeries], config: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    grouped: dict[str, list[SourceSeries]] = defaultdict(list)
    for record in records:
        grouped[record.source_group_id].append(record)
    assignments: dict[str, str] = {}
    reasons: dict[str, str] = {}
    unresolved: list[str] = []
    for group_id, members in sorted(grouped.items()):
        official = {member.official_split for member in members}
        if "evaluation" in official:
            assignments[group_id] = "test"
            reasons[group_id] = "group contains official TSB-AD evaluation file"
        elif "tuning" in official:
            assignments[group_id] = "validation"
            reasons[group_id] = "group contains official TSB-AD tuning file"
        else:
            unresolved.append(group_id)

    development_groups = [group for group in grouped if assignments.get(group) != "test"]
    validation_fraction = float(config["validation_fraction_of_development_groups"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction_of_development_groups must be in (0, 1)")
    validation_target = max(1, int(math.ceil(len(development_groups) * validation_fraction)))
    fixed_validation = sum(assignments.get(group) == "validation" for group in grouped)
    ordered = sorted(
        unresolved,
        key=lambda group: hashlib.sha256(
            f"{int(config['seed'])}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    extra_validation = max(0, validation_target - fixed_validation)
    for index, group_id in enumerate(ordered):
        if index < extra_validation:
            assignments[group_id] = "validation"
            reasons[group_id] = "deterministic validation fill at source-group level"
        else:
            assignments[group_id] = "train"
            reasons[group_id] = "deterministic development-group allocation"
    return assignments, reasons


def _event_windows(
    record: SourceSeries,
    split: str,
    window_length: int,
    max_event_duration: int,
) -> list[WindowRecord]:
    result: list[WindowRecord] = []
    if len(record.signal) < window_length:
        return result
    for event_index, (event_start, event_end) in enumerate(event_intervals(record.labels)):
        if event_end - event_start > max_event_duration:
            continue
        center = (event_start + event_end - 1) // 2
        start = max(0, min(center - window_length // 2, len(record.signal) - window_length))
        end = start + window_length
        result.append(
            WindowRecord(
                split=split,
                window_id=f"{record.parsed.file_name}:event:{event_index}:{start}:{end}",
                series_id=record.parsed.file_name,
                source_group_id=record.source_group_id,
                archive_member=record.member,
                kind="event",
                start=start,
                end=end,
                event_index=event_index,
                signal=record.signal[start:end].astype(np.float32),
                labels=record.labels[start:end].astype(np.uint8),
            )
        )
    return result


def _append_one_normal_window_per_group(
    records: Sequence[SourceSeries],
    assignments: Mapping[str, str],
    destination: list[WindowRecord],
    window_length: int,
) -> None:
    groups: dict[str, list[SourceSeries]] = defaultdict(list)
    for record in records:
        groups[record.source_group_id].append(record)
    for group_id, members in sorted(groups.items()):
        record = min(members, key=lambda item: item.parsed.file_name)
        if record.parsed.training_end < window_length:
            continue
        available = record.parsed.training_end - window_length + 1
        offset = int(record.normal_prefix_sha256[:16], 16) % available
        end = offset + window_length
        labels = record.labels[offset:end]
        if np.any(labels):
            raise AssertionError("Normal-prefix window unexpectedly contains an anomaly label")
        split = assignments[group_id]
        destination.append(
            WindowRecord(
                split=split,
                window_id=f"{record.parsed.file_name}:normal:{offset}:{end}",
                series_id=record.parsed.file_name,
                source_group_id=group_id,
                archive_member=record.member,
                kind="normal",
                start=offset,
                end=end,
                event_index=None,
                signal=record.signal[offset:end].astype(np.float32),
                labels=labels.astype(np.uint8),
            )
        )


def _source_manifest(
    record: SourceSeries,
    split: str,
    reason: str,
    windows: Sequence[WindowRecord],
    max_event_duration: int,
) -> dict[str, Any]:
    events = event_intervals(record.labels)
    return {
        "archive_member": record.member,
        "file_name": record.parsed.file_name,
        "file_sha256": record.file_sha256,
        "global_id": record.parsed.global_id,
        "source_collection": record.parsed.source_collection,
        "source_id": record.parsed.source_id,
        "domain": record.parsed.domain,
        "samples": len(record.signal),
        "training_prefix_end": record.parsed.training_end,
        "training_prefix_positive_labels": int(
            record.labels[: record.parsed.training_end].sum()
        ),
        "first_anomaly_index_from_filename": record.parsed.first_anomaly,
        "normal_prefix_sha256": record.normal_prefix_sha256,
        "source_group_id": record.source_group_id,
        "official_split": record.official_split,
        "assigned_split": split,
        "assignment_reason": reason,
        "events": [{"start": start, "end": end} for start, end in events],
        "eligible_event_windows": len(windows),
        "excluded_events_over_duration_limit": sum(
            end - start > max_event_duration for start, end in events
        ),
    }


def _assert_group_disjoint(split_groups: Mapping[str, Iterable[str]]) -> None:
    names = list(split_groups)
    sets = {name: set(split_groups[name]) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(
                    f"Source-group leakage between {left} and {right}: {sorted(overlap)[:3]}"
                )


def _summary(
    records: Sequence[Mapping[str, Any]], windows: Sequence[WindowRecord]
) -> dict[str, Any]:
    splits = ("train", "validation", "test", "transfer")
    return {
        "sources_by_split": {
            split: sum(record["assigned_split"] == split for record in records)
            for split in splits
        },
        "groups_by_split": {
            split: len(
                {
                    record["source_group_id"]
                    for record in records
                    if record["assigned_split"] == split
                }
            )
            for split in splits
        },
        "windows_by_split": {
            split: sum(window.split == split for window in windows) for split in splits
        },
        "event_windows_by_split": {
            split: sum(window.split == split and window.kind == "event" for window in windows)
            for split in splits
        },
        "normal_windows_by_split": {
            split: sum(window.split == split and window.kind == "normal" for window in windows)
            for split in splits
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_expected_sha256(path: Path, expected: Any, description: str) -> None:
    if expected is None:
        return
    actual = _sha256_file(path)
    if actual.casefold() != str(expected).strip().casefold():
        raise ValueError(
            f"SHA-256 mismatch for {description} {path}: "
            f"expected {expected}, got {actual}"
        )
