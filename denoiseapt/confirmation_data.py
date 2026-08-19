"""Build and verify the sealed protocol-v2 confirmation data artifact.

This module is deliberately data-only.  It never imports the model, training,
inference, metric, or experiment-runner modules.  It uses labels solely to
predefine complete event windows and construction counts.

Leakage controls are stronger than protocol v1:

* only target-domain files on the pinned official evaluation list are eligible;
* an exact normal-prefix group is quarantined if any member is a tuning or
  otherwise non-evaluation file, has a contaminated declared prefix, is too
  short, or matches a protocol-v1 normal-prefix hash;
* a fixed number of groups per domain are reserved for calibration before any
  confirmation evaluation; and
* calibration windows contain only samples before the filename ``tr_``
  boundary.  No source group can occur in both calibration and confirmation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from numpy.typing import NDArray

from denoiseapt.experiment_data import (
    ParsedName,
    _read_tsb_csv,
    event_intervals,
    normal_prefix_sha256,
    parse_tsb_ad_filename,
    read_official_file_list,
)


_ARRAY_HASH_DOMAIN = b"DenoiseAPT-confirmation-array-v1\0"
_SELECTION_HASH_DOMAIN = b"DenoiseAPT-confirmation-selection-v1\0"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ConfirmationSource:
    member: str
    parsed: ParsedName
    signal: NDArray[np.float64]
    labels: NDArray[np.uint8]
    file_sha256: str
    normal_prefix_sha256: str
    source_group_id: str
    official_split: str
    prefix_positive_labels: int


@dataclass(frozen=True)
class ConfirmationWindow:
    domain_key: str
    partition: str
    kind: str
    window_id: str
    series_id: str
    source_group_id: str
    archive_member: str
    start: int
    end: int
    signal: NDArray[np.float32]
    labels: NDArray[np.uint8]
    event_index: int | None = None
    target_event_start: int | None = None
    target_event_end: int | None = None

    def manifest_dict(self) -> dict[str, Any]:
        target_start = self.target_event_start
        target_end = self.target_event_end
        return {
            "domain_key": self.domain_key,
            "partition": self.partition,
            "kind": self.kind,
            "window_id": self.window_id,
            "series_id": self.series_id,
            "source_group_id": self.source_group_id,
            "archive_member": self.archive_member,
            "start": self.start,
            "end": self.end,
            "event_index": self.event_index,
            "target_event_start": target_start,
            "target_event_end": target_end,
            "target_event_duration": (
                None
                if target_start is None or target_end is None
                else target_end - target_start
            ),
            "target_event_offset": (
                None
                if target_start is None or target_end is None
                else [target_start - self.start, target_end - self.start]
            ),
            "positive_samples": int(self.labels.sum()),
            "normal_samples": int((self.labels == 0).sum()),
            "signal_sha256": canonical_array_sha256(self.signal),
            "labels_sha256": canonical_array_sha256(self.labels),
        }


def canonical_array_sha256(values: np.ndarray) -> str:
    """Hash a numeric array with explicit dtype and shape framing."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("Object arrays are not permitted in confirmation artifacts")
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(_ARRAY_HASH_DOMAIN)
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def deterministic_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with fixed member order, timestamps, and permissions."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"Unsafe NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"Object array is forbidden: {name}")
            stream = io.BytesIO()
            np.lib.format.write_array(stream, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            bundle.writestr(info, stream.getvalue(), compresslevel=9)
    temporary.replace(destination)


def build_confirmation_artifacts(
    *,
    archive_path: str | Path,
    protocol_config: str | Path | Mapping[str, Any],
    output_npz: str | Path,
    output_manifest: str | Path,
    output_lock: str | Path,
) -> dict[str, Any]:
    """Construct a deterministic, group-disjoint confirmation artifact.

    A pre-existing lock makes the files immutable: the function verifies and
    returns them without rebuilding.  Any change requires a new protocol id and
    new output paths.
    """

    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_npz).expanduser().resolve()
    manifest_path = Path(output_manifest).expanduser().resolve()
    lock_path = Path(output_lock).expanduser().resolve()
    config, config_path = _load_config(protocol_config)
    base_dir = config_path.parent if config_path is not None else Path.cwd()

    if lock_path.exists():
        verify_confirmation_lock(
            archive_path=archive,
            protocol_config=protocol_config,
            output_npz=output,
            output_manifest=manifest_path,
            output_lock=lock_path,
        )
        return json.loads(manifest_path.read_text("utf-8"))
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an unlocked confirmation artifact: "
            + ", ".join(str(path) for path in existing)
        )

    _validate_config(config)
    expected_archive = config["archive"]
    _verify_file(
        archive,
        expected_sha256=expected_archive["sha256"],
        expected_bytes=expected_archive.get("bytes"),
        description="pinned TSB-AD-U archive",
    )

    official = config["official_lists"]
    eval_path = (base_dir / official["evaluation"]).resolve()
    tuning_path = (base_dir / official["tuning"]).resolve()
    _verify_file(
        eval_path,
        expected_sha256=official["evaluation_sha256"],
        expected_bytes=None,
        description="official evaluation list",
    )
    _verify_file(
        tuning_path,
        expected_sha256=official["tuning_sha256"],
        expected_bytes=None,
        description="official tuning list",
    )
    eval_names = read_official_file_list(eval_path)
    tuning_names = read_official_file_list(tuning_path)
    if len(eval_names) != int(official["evaluation_entries"]):
        raise ValueError("Official evaluation entry count differs from frozen config")
    if len(tuning_names) != int(official["tuning_entries"]):
        raise ValueError("Official tuning entry count differs from frozen config")
    if eval_names & tuning_names:
        raise ValueError("Pinned official evaluation and tuning lists overlap")

    prior = config["prior_spent_protocol"]
    prior_path = (base_dir / prior["manifest"]).resolve()
    _verify_file(
        prior_path,
        expected_sha256=prior["manifest_sha256"],
        expected_bytes=None,
        description="spent protocol-v1 manifest",
    )
    prior_manifest = json.loads(prior_path.read_text("utf-8"))
    prior_prefix_hashes = {
        item["normal_prefix_sha256"] for item in prior_manifest["sources"]
    }

    domain_specs = {item["key"]: item for item in config["domains"]}
    records_by_domain: dict[str, list[ConfirmationSource]] = {
        key: [] for key in domain_specs
    }
    with zipfile.ZipFile(archive) as bundle:
        for info in sorted(bundle.infolist(), key=lambda item: item.filename):
            if info.is_dir() or not info.filename.casefold().endswith(".csv"):
                continue
            try:
                parsed = parse_tsb_ad_filename(Path(info.filename).name)
            except ValueError:
                continue
            domain_key = next(
                (
                    key
                    for key, spec in domain_specs.items()
                    if parsed.source_collection == spec["source_collection"]
                    and parsed.domain == spec["domain"]
                ),
                None,
            )
            if domain_key is None:
                continue
            raw = bundle.read(info)
            signal, labels = _read_tsb_csv(raw, parsed.file_name)
            if parsed.training_end > len(signal):
                raise ValueError(
                    f"Filename training boundary exceeds series: {parsed.file_name}"
                )
            prefix_hash = normal_prefix_sha256(signal, parsed.training_end)
            role = (
                "evaluation"
                if parsed.file_name in eval_names
                else "tuning"
                if parsed.file_name in tuning_names
                else "unlisted"
            )
            records_by_domain[domain_key].append(
                ConfirmationSource(
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
                    official_split=role,
                    prefix_positive_labels=int(labels[: parsed.training_end].sum()),
                )
            )

    window_length = int(config["window_length"])
    max_event_duration = int(config["max_event_duration"])
    min_event_normal = int(config["minimum_normal_samples_in_event_window"])
    calibration_count = int(config["calibration_groups_per_domain"])
    minimum_confirmation = int(config["minimum_confirmation_groups_per_domain"])
    all_windows: list[ConfirmationWindow] = []
    source_inventory: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    domain_manifests: dict[str, dict[str, Any]] = {}

    for domain_key, spec in domain_specs.items():
        records = records_by_domain[domain_key]
        if not records:
            raise ValueError(f"No archive sources found for domain {domain_key!r}")
        groups: dict[str, list[ConfirmationSource]] = defaultdict(list)
        for record in records:
            groups[record.source_group_id].append(record)

        eligible: dict[str, list[ConfirmationSource]] = {}
        group_reasons: dict[str, str] = {}
        for group_id, members in sorted(groups.items()):
            reasons: list[str] = []
            roles = {member.official_split for member in members}
            if roles != {"evaluation"}:
                reasons.append("group_not_exclusively_official_evaluation")
            if any(member.prefix_positive_labels for member in members):
                reasons.append("declared_normal_prefix_contains_positive_labels")
            if any(
                len(member.signal) < window_length
                or member.parsed.training_end < window_length
                for member in members
            ):
                reasons.append("insufficient_samples_for_fixed_window")
            if any(
                member.normal_prefix_sha256 in prior_prefix_hashes for member in members
            ):
                reasons.append("exact_normal_prefix_seen_in_spent_protocol_v1")
            if reasons:
                reason = ";".join(reasons)
                group_reasons[group_id] = reason
                exclusions.append(
                    {
                        "domain_key": domain_key,
                        "scope": "source_group",
                        "source_group_id": group_id,
                        "reason": reason,
                        "members": sorted(member.parsed.file_name for member in members),
                    }
                )
            else:
                eligible[group_id] = members

        if len(eligible) < calibration_count + minimum_confirmation:
            raise ValueError(
                f"Domain {domain_key!r} has {len(eligible)} eligible groups; "
                f"needs at least {calibration_count + minimum_confirmation}"
            )
        calibration_groups = sorted(
            eligible,
            key=lambda group_id: _calibration_rank(
                int(config["seed"]), domain_key, group_id
            ),
        )[:calibration_count]
        confirmation_groups = sorted(set(eligible) - set(calibration_groups))
        if len(confirmation_groups) < minimum_confirmation:
            raise AssertionError("Confirmation-group minimum was not preserved")

        partition_by_group = {
            **{group: "calibration" for group in calibration_groups},
            **{group: "confirmation" for group in confirmation_groups},
        }
        for group_id, members in sorted(groups.items()):
            partition = partition_by_group.get(group_id, "excluded")
            reason = (
                "normal-prefix-only detector threshold calibration"
                if partition == "calibration"
                else "sealed confirmation evaluation"
                if partition == "confirmation"
                else group_reasons[group_id]
            )
            for record in sorted(members, key=lambda item: item.parsed.file_name):
                source_inventory.append(
                    _source_manifest(record, domain_key, partition, reason)
                )

        calibration_windows: list[ConfirmationWindow] = []
        for group_id in calibration_groups:
            calibration_windows.append(
                _normal_prefix_window(
                    domain_key=domain_key,
                    partition="calibration",
                    members=eligible[group_id],
                    seed=int(config["seed"]),
                    window_length=window_length,
                )
            )

        confirmation_windows: list[ConfirmationWindow] = []
        excluded_long_events = 0
        for group_id in confirmation_groups:
            members = eligible[group_id]
            confirmation_windows.append(
                _normal_prefix_window(
                    domain_key=domain_key,
                    partition="confirmation",
                    members=members,
                    seed=int(config["seed"]),
                    window_length=window_length,
                )
            )
            for record in sorted(members, key=lambda item: item.parsed.file_name):
                for event_index, (event_start, event_end) in enumerate(
                    event_intervals(record.labels)
                ):
                    duration = event_end - event_start
                    if duration > max_event_duration:
                        excluded_long_events += 1
                        exclusions.append(
                            {
                                "domain_key": domain_key,
                                "scope": "event",
                                "series_id": record.parsed.file_name,
                                "event_index": event_index,
                                "event_start": event_start,
                                "event_end": event_end,
                                "event_duration": duration,
                                "reason": "event_duration_exceeds_frozen_limit",
                            }
                        )
                        continue
                    window = _event_window(
                        domain_key=domain_key,
                        record=record,
                        event_index=event_index,
                        event_start=event_start,
                        event_end=event_end,
                        window_length=window_length,
                    )
                    if int((window.labels == 0).sum()) < min_event_normal:
                        raise ValueError(
                            f"Event window has fewer than {min_event_normal} normal "
                            f"samples: {window.window_id}"
                        )
                    confirmation_windows.append(window)

        if set(calibration_groups) & set(confirmation_groups):
            raise AssertionError("Calibration and confirmation groups overlap")
        all_windows.extend(calibration_windows)
        all_windows.extend(confirmation_windows)
        selection = {
            "eligible_groups": sorted(eligible),
            "calibration_groups": calibration_groups,
            "confirmation_groups": confirmation_groups,
        }
        source_counts = {
            partition: sum(
                len(eligible[group_id])
                for group_id in (
                    calibration_groups
                    if partition == "calibration"
                    else confirmation_groups
                )
            )
            for partition in ("calibration", "confirmation")
        }
        domain_manifests[domain_key] = {
            "source_collection": spec["source_collection"],
            "domain": spec["domain"],
            "confirmation_role": spec["confirmation_role"],
            "selection_sha256": _selection_sha256(selection),
            "selection": selection,
            "counts": {
                "archive_sources": len(records),
                "archive_groups": len(groups),
                "eligible_sources": sum(len(items) for items in eligible.values()),
                "eligible_groups": len(eligible),
                "calibration_sources": source_counts["calibration"],
                "calibration_groups": len(calibration_groups),
                "calibration_windows": len(calibration_windows),
                "confirmation_sources": source_counts["confirmation"],
                "confirmation_groups": len(confirmation_groups),
                "confirmation_event_windows": sum(
                    window.kind == "event" for window in confirmation_windows
                ),
                "confirmation_normal_windows": sum(
                    window.kind == "normal" for window in confirmation_windows
                ),
                "confirmation_windows": len(confirmation_windows),
                "events_excluded_over_duration_limit": excluded_long_events,
                "quarantined_groups": len(groups) - len(eligible),
            },
        }

    _assert_partition_disjointness(all_windows)
    forbidden = {
        (item["source_collection"], item["domain"])
        for item in config["forbidden_spent_domains"]
    }
    for source in source_inventory:
        if (source["source_collection"], source["domain"]) in forbidden or (
            source["source_collection"], "*"
        ) in forbidden:
            raise AssertionError("A spent domain entered confirmation inventory")

    manifest: dict[str, Any] = {
        "schema_version": int(config["schema_version"]),
        "protocol_id": config["protocol_id"],
        "status": "SEALED_DATA_ONLY_NO_MODEL_EVALUATION",
        "sealed_utc_date": config["sealed_utc_date"],
        "seed": int(config["seed"]),
        "window_length": window_length,
        "max_event_duration": max_event_duration,
        "minimum_normal_samples_in_event_window": min_event_normal,
        "archive": {
            "file_name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": _sha256_file(archive),
            "url": expected_archive["url"],
        },
        "official_lists": {
            "repository_commit": official["repository_commit"],
            "evaluation": {
                "file_name": eval_path.name,
                "entries": len(eval_names),
                "sha256": _sha256_file(eval_path),
                "url": official["evaluation_url"],
            },
            "tuning": {
                "file_name": tuning_path.name,
                "entries": len(tuning_names),
                "sha256": _sha256_file(tuning_path),
                "url": official["tuning_url"],
            },
        },
        "prior_spent_protocol": {
            "protocol_id": prior_manifest["protocol_id"],
            "manifest_file_name": prior_path.name,
            "manifest_sha256": _sha256_file(prior_path),
            "normal_prefix_hashes_compared": len(prior_prefix_hashes),
            "exact_prefix_overlap_with_confirmation": False,
        },
        "forbidden_spent_domains": config["forbidden_spent_domains"],
        "grouping": {
            "algorithm": "sha256-domain-separated-float64le-normal-prefix-v1",
            "calibration_assignment": (
                "per-domain ascending sha256(seed:calibration:domain_key:group_id), "
                "first frozen count"
            ),
            "whole_group_partitioning": True,
            "calibration_confirmation_group_disjointness_verified": True,
            "source_overlap_policy": config["source_overlap_policy"],
        },
        "construction_only_notice": config["construction_only_notice"],
        "provenance_and_licenses": config["provenance_and_licenses"],
        "domains": domain_manifests,
        "summary": _summary(domain_manifests),
        "sources": sorted(
            source_inventory,
            key=lambda item: (item["domain_key"], item["file_name"]),
        ),
        "exclusions": sorted(exclusions, key=_exclusion_sort_key),
        "windows": [
            window.manifest_dict()
            for window in sorted(all_windows, key=lambda item: item.window_id)
        ],
    }

    arrays = _artifact_arrays(all_windows, manifest)
    deterministic_npz(output, arrays)
    manifest["artifact"] = {
        "file_name": output.name,
        "bytes": output.stat().st_size,
        "sha256": _sha256_file(output),
        "arrays": sorted(arrays),
        "deterministic_container": (
            "NPZ/DEFLATE level 9; lexicographic members; fixed 1980-01-01 ZIP "
            "timestamps; NumPy NPY allow_pickle=False"
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    implementation_hashes: dict[str, dict[str, Any]] = {}
    for relative in config.get("implementation_files", []):
        implementation_path = (base_dir / relative).resolve()
        implementation_hashes[relative] = {
            "file_name": implementation_path.name,
            "sha256": _sha256_file(implementation_path),
        }
    lock = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "SEALED_BEFORE_MODEL_EVALUATION",
        "sealed_utc_date": config["sealed_utc_date"],
        "immutable_policy": (
            "Do not replace the config, NPZ, manifest, group assignment, or lock. "
            "Any change requires a newly named protocol and fresh lock."
        ),
        "permitted_future_use": (
            "Apply already-frozen models and detector definitions exactly once under "
            "a preregistered evaluation runner; calibration may use only calibration "
            "arrays; confirmation arrays may not tune any parameter."
        ),
        "prohibited_use": [
            "model selection or hyperparameter tuning on confirmation arrays",
            "threshold calibration on confirmation arrays",
            "using calibration-group post-boundary samples or labels",
            "merging UCR-Medical or SMAP into this confirmation protocol",
            "editing or replacing sealed files in place",
        ],
        "construction_attestation": (
            "No DenoiseAPT/controller model, checkpoint, inference trace, metric, or "
            "outcome summary was loaded or produced by this builder."
        ),
        "hashes": {
            "archive": _sha256_file(archive),
            "config": _config_sha256(config, config_path),
            "official_evaluation_list": _sha256_file(eval_path),
            "official_tuning_list": _sha256_file(tuning_path),
            "prior_spent_protocol_manifest": _sha256_file(prior_path),
            "artifact_npz": _sha256_file(output),
            "manifest": _sha256_file(manifest_path),
            "implementation_files": implementation_hashes,
        },
        "domain_selection_sha256": {
            key: value["selection_sha256"] for key, value in domain_manifests.items()
        },
        "construction_counts": manifest["summary"],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify_confirmation_lock(
        archive_path=archive,
        protocol_config=protocol_config,
        output_npz=output,
        output_manifest=manifest_path,
        output_lock=lock_path,
    )
    return manifest


def verify_confirmation_lock(
    *,
    archive_path: str | Path,
    protocol_config: str | Path | Mapping[str, Any],
    output_npz: str | Path,
    output_manifest: str | Path,
    output_lock: str | Path,
) -> dict[str, Any]:
    """Verify all sealed hashes without loading any model or outcome artifact."""

    archive = Path(archive_path).expanduser().resolve()
    output = Path(output_npz).expanduser().resolve()
    manifest_path = Path(output_manifest).expanduser().resolve()
    lock_path = Path(output_lock).expanduser().resolve()
    config, config_path = _load_config(protocol_config)
    lock = json.loads(lock_path.read_text("utf-8"))
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if lock.get("status") != "SEALED_BEFORE_MODEL_EVALUATION":
        raise ValueError("Confirmation lock is not in the sealed state")
    if lock.get("protocol_id") != config.get("protocol_id"):
        raise ValueError("Confirmation lock protocol id differs from config")
    expected = lock["hashes"]
    checks = {
        "archive": _sha256_file(archive),
        "config": _config_sha256(config, config_path),
        "artifact_npz": _sha256_file(output),
        "manifest": _sha256_file(manifest_path),
    }
    base_dir = config_path.parent if config_path is not None else Path.cwd()
    official = config["official_lists"]
    prior = config["prior_spent_protocol"]
    checks.update(
        {
            "official_evaluation_list": _sha256_file(
                (base_dir / official["evaluation"]).resolve()
            ),
            "official_tuning_list": _sha256_file(
                (base_dir / official["tuning"]).resolve()
            ),
            "prior_spent_protocol_manifest": _sha256_file(
                (base_dir / prior["manifest"]).resolve()
            ),
        }
    )
    for name, actual in checks.items():
        if actual != expected[name]:
            raise ValueError(
                f"Sealed confirmation {name} SHA-256 mismatch: "
                f"expected {expected[name]}, got {actual}"
            )
    for relative, frozen in expected.get("implementation_files", {}).items():
        actual = _sha256_file((base_dir / relative).resolve())
        if actual != frozen["sha256"]:
            raise ValueError(
                f"Sealed confirmation implementation SHA-256 mismatch for {relative}: "
                f"expected {frozen['sha256']}, got {actual}"
            )
    if manifest.get("artifact", {}).get("sha256") != expected["artifact_npz"]:
        raise ValueError("Manifest artifact hash differs from confirmation lock")
    for key, selection_hash in lock["domain_selection_sha256"].items():
        if manifest["domains"][key]["selection_sha256"] != selection_hash:
            raise ValueError(f"Selection hash mismatch for domain {key}")
    calibration = {
        window["source_group_id"]
        for window in manifest["windows"]
        if window["partition"] == "calibration"
    }
    confirmation = {
        window["source_group_id"]
        for window in manifest["windows"]
        if window["partition"] == "confirmation"
    }
    overlap = calibration & confirmation
    if overlap:
        raise ValueError(f"Sealed manifest has cross-partition groups: {sorted(overlap)}")
    if any(
        window["partition"] == "calibration"
        and (window["kind"] != "normal" or window["positive_samples"] != 0)
        for window in manifest["windows"]
    ):
        raise ValueError("Calibration partition contains non-normal windows")
    return lock


def _normal_prefix_window(
    *,
    domain_key: str,
    partition: str,
    members: Sequence[ConfirmationSource],
    seed: int,
    window_length: int,
) -> ConfirmationWindow:
    record = min(members, key=lambda item: item.parsed.file_name)
    available = record.parsed.training_end - window_length + 1
    if available <= 0:
        raise ValueError("Normal prefix cannot provide the frozen window length")
    token = (
        f"{seed}:normal-prefix-window:{domain_key}:{partition}:"
        f"{record.source_group_id}"
    )
    offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16) % available
    end = offset + window_length
    labels = record.labels[offset:end]
    if np.any(labels):
        raise AssertionError("Declared normal-prefix window contains positive labels")
    if end > record.parsed.training_end:
        raise AssertionError("Normal-prefix window crossed the training boundary")
    return ConfirmationWindow(
        domain_key=domain_key,
        partition=partition,
        kind="normal",
        window_id=(
            f"{domain_key}:{partition}:{record.parsed.file_name}:"
            f"normal:{offset}:{end}"
        ),
        series_id=record.parsed.file_name,
        source_group_id=record.source_group_id,
        archive_member=record.member,
        start=offset,
        end=end,
        signal=record.signal[offset:end].astype(np.float32),
        labels=labels.astype(np.uint8),
    )


def _event_window(
    *,
    domain_key: str,
    record: ConfirmationSource,
    event_index: int,
    event_start: int,
    event_end: int,
    window_length: int,
) -> ConfirmationWindow:
    center = (event_start + event_end - 1) // 2
    start = max(0, min(center - window_length // 2, len(record.signal) - window_length))
    end = start + window_length
    if not start <= event_start < event_end <= end:
        raise ValueError("Frozen event window does not contain the complete target event")
    return ConfirmationWindow(
        domain_key=domain_key,
        partition="confirmation",
        kind="event",
        window_id=(
            f"{domain_key}:confirmation:{record.parsed.file_name}:"
            f"event:{event_index}:{start}:{end}"
        ),
        series_id=record.parsed.file_name,
        source_group_id=record.source_group_id,
        archive_member=record.member,
        start=start,
        end=end,
        signal=record.signal[start:end].astype(np.float32),
        labels=record.labels[start:end].astype(np.uint8),
        event_index=event_index,
        target_event_start=event_start,
        target_event_end=event_end,
    )


def _source_manifest(
    record: ConfirmationSource,
    domain_key: str,
    partition: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "domain_key": domain_key,
        "archive_member": record.member,
        "file_name": record.parsed.file_name,
        "file_sha256": record.file_sha256,
        "global_id": record.parsed.global_id,
        "source_collection": record.parsed.source_collection,
        "source_id": record.parsed.source_id,
        "domain": record.parsed.domain,
        "samples": len(record.signal),
        "training_prefix_end": record.parsed.training_end,
        "training_prefix_positive_labels": record.prefix_positive_labels,
        "first_anomaly_index_from_filename": record.parsed.first_anomaly,
        "normal_prefix_sha256": record.normal_prefix_sha256,
        "source_group_id": record.source_group_id,
        "official_split": record.official_split,
        "partition": partition,
        "partition_reason": reason,
    }


def _artifact_arrays(
    windows: Sequence[ConfirmationWindow], manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    metadata = {
        "schema_version": manifest["schema_version"],
        "protocol_id": manifest["protocol_id"],
        "status": manifest["status"],
        "window_length": manifest["window_length"],
        "raw_reference_values": True,
        "normalised": False,
        "calibration_is_normal_prefix_only": True,
        "calibration_confirmation_group_disjoint": True,
        "construction_counts": manifest["summary"],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True))
    }
    for domain_key in sorted(manifest["domains"]):
        for partition in ("calibration", "confirmation"):
            selected = sorted(
                (
                    window
                    for window in windows
                    if window.domain_key == domain_key
                    and window.partition == partition
                ),
                key=lambda item: item.window_id,
            )
            prefix = f"{domain_key}_{partition}"
            arrays[f"{prefix}_clean"] = np.stack(
                [window.signal for window in selected]
            ).astype(np.float32)
            arrays[f"{prefix}_labels"] = np.stack(
                [window.labels for window in selected]
            ).astype(np.uint8)
            arrays[f"{prefix}_kind"] = np.asarray(
                [window.kind for window in selected]
            )
            arrays[f"{prefix}_domain_key"] = np.asarray(
                [domain_key for _ in selected]
            )
            source_domain = (
                f"{manifest['domains'][domain_key]['source_collection']}/"
                f"{manifest['domains'][domain_key]['domain']}"
            )
            arrays[f"{prefix}_source_domain"] = np.asarray(
                [source_domain for _ in selected]
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
            arrays[f"{prefix}_event_index"] = np.asarray(
                [-1 if window.event_index is None else window.event_index for window in selected],
                dtype=np.int32,
            )
            arrays[f"{prefix}_target_event_start"] = np.asarray(
                [
                    -1
                    if window.target_event_start is None
                    else window.target_event_start
                    for window in selected
                ],
                dtype=np.int64,
            )
            arrays[f"{prefix}_target_event_end"] = np.asarray(
                [
                    -1 if window.target_event_end is None else window.target_event_end
                    for window in selected
                ],
                dtype=np.int64,
            )
    return arrays


def _assert_partition_disjointness(windows: Iterable[ConfirmationWindow]) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for window in windows:
        groups[window.partition].add(window.source_group_id)
        if window.window_id in ids:
            raise AssertionError(f"Duplicate confirmation window id: {window.window_id}")
        ids.add(window.window_id)
    overlap = groups["calibration"] & groups["confirmation"]
    if overlap:
        raise AssertionError(
            f"Calibration/confirmation source-group leakage: {sorted(overlap)[:3]}"
        )


def _summary(domains: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "calibration_sources",
        "calibration_groups",
        "calibration_windows",
        "confirmation_sources",
        "confirmation_groups",
        "confirmation_event_windows",
        "confirmation_normal_windows",
        "confirmation_windows",
        "quarantined_groups",
    )
    return {
        "by_domain": {
            key: dict(value["counts"]) for key, value in sorted(domains.items())
        },
        "total": {
            name: sum(int(value["counts"][name]) for value in domains.values())
            for name in keys
        },
    }


def _calibration_rank(seed: int, domain_key: str, group_id: str) -> str:
    return hashlib.sha256(
        f"{seed}:calibration:{domain_key}:{group_id}".encode("utf-8")
    ).hexdigest()


def _selection_sha256(selection: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_SELECTION_HASH_DOMAIN)
    digest.update(
        json.dumps(selection, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _exclusion_sort_key(item: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(item.get("domain_key", "")),
        str(item.get("scope", "")),
        str(item.get("source_group_id", "")),
        str(item.get("series_id", "")),
        str(item.get("event_index", "")),
    )


def _load_config(
    value: str | Path | Mapping[str, Any]
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value).expanduser().resolve()
    return json.loads(path.read_text("utf-8")), path


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config["window_length"]) <= 1:
        raise ValueError("window_length must exceed one sample")
    if not 0 < int(config["max_event_duration"]) <= int(config["window_length"]):
        raise ValueError("max_event_duration must be in (0, window_length]")
    if int(config["minimum_normal_samples_in_event_window"]) < 1:
        raise ValueError("Event windows must retain at least one normal sample")
    if int(config["calibration_groups_per_domain"]) < 1:
        raise ValueError("At least one calibration group per domain is required")
    domain_keys = [item["key"] for item in config["domains"]]
    if len(domain_keys) != len(set(domain_keys)):
        raise ValueError("Confirmation domain keys must be unique")
    forbidden = {
        (item["source_collection"], item["domain"])
        for item in config["forbidden_spent_domains"]
    }
    for domain in config["domains"]:
        identity = (domain["source_collection"], domain["domain"])
        if identity in forbidden or (identity[0], "*") in forbidden:
            raise ValueError(f"A target domain is marked spent: {identity}")


def _config_sha256(config: Mapping[str, Any], path: Path | None) -> str:
    if path is not None:
        return _sha256_file(path)
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    description: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(
            f"Byte-size mismatch for {description}: expected {expected_bytes}, "
            f"got {path.stat().st_size}"
        )
    actual = _sha256_file(path)
    if actual.casefold() != str(expected_sha256).strip().casefold():
        raise ValueError(
            f"SHA-256 mismatch for {description}: expected {expected_sha256}, got {actual}"
        )
