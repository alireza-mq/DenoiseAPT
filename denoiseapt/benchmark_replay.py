"""Integrity-checked replay of one frozen held-out benchmark window.

The replay keeps the public-facing demonstration honest about what is live and
what is already computed.  External comparator outputs are loaded from the
audited matched-panel artifacts; they are not presented as browser-time model
executions.  Expert edits operate only on the displayed DenoiseAPT output and
the corresponding corrupted observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


SERIES_KEYS = (
    "observation",
    "median_filter_w3",
    "wavelet_shrinkage",
    "noisereduce",
    "rins_t",
    "our_model",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vector(value: Any, name: str, *, length: int | None = None) -> NDArray:
    array = np.asarray(value).reshape(-1)
    if length is not None and array.size != length:
        raise ValueError(f"{name} must contain {length} values")
    return array


@dataclass(frozen=True)
class BenchmarkReplayCase:
    """One base window and its frozen matched corruption conditions."""

    path: Path
    metadata: dict[str, Any]
    reference: NDArray[np.float32]
    labels: NDArray[np.bool_]
    condition_id: NDArray[np.str_]
    family: NDArray[np.str_]
    severity: NDArray[np.float32]
    replicate: NDArray[np.int32]
    identity: NDArray[np.bool_]
    center: NDArray[np.float32]
    scale: NDArray[np.float32]
    series: dict[str, NDArray[np.float32]]

    @property
    def length(self) -> int:
        return int(self.reference.size)

    @property
    def default_index(self) -> int:
        target = str(self.metadata["default_condition_id"])
        matches = np.flatnonzero(self.condition_id == target)
        if matches.size != 1:
            raise ValueError("The default replay condition is missing or duplicated")
        return int(matches[0])

    def condition_index(self, family: str, severity: float, replicate: int) -> int:
        if family == "none" or float(severity) == 0.0:
            matches = np.flatnonzero(self.identity)
        else:
            matches = np.flatnonzero(
                (~self.identity)
                & (self.family == str(family))
                & np.isclose(self.severity, float(severity), rtol=0.0, atol=1e-7)
                & (self.replicate == int(replicate))
            )
        if matches.size != 1:
            raise ValueError(
                "The selected family, severity, and replicate are not in the frozen replay grid"
            )
        return int(matches[0])


def load_benchmark_replay(path: Path) -> BenchmarkReplayCase:
    """Load a replay artifact and fail closed on schema or numeric drift."""

    manifest_path = path.with_name(f"{path.stem}_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Replay integrity manifest is missing or invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or Path(str(manifest.get("artifact", ""))).name != path.name
        or str(manifest.get("artifact_sha256", "")).lower() != _sha256(path)
    ):
        raise ValueError("Replay artifact does not match its integrity manifest")

    with np.load(path, allow_pickle=False) as item:
        required = {
            "metadata_json",
            "reference",
            "labels",
            "condition_id",
            "family",
            "severity",
            "replicate",
            "identity",
            "center",
            "scale",
            *SERIES_KEYS,
        }
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"Replay artifact is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(item["metadata_json"].item()))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
            raise ValueError("Replay metadata must use schema version 1")
        reference = np.asarray(item["reference"], dtype=np.float32).reshape(-1)
        if reference.size != 512 or not np.isfinite(reference).all():
            raise ValueError("Replay reference must be one finite 512-point window")
        labels = np.asarray(item["labels"], dtype=bool).reshape(-1)
        if labels.size != reference.size:
            raise ValueError("Replay labels must align with the reference")
        condition_id = _vector(item["condition_id"], "condition_id")
        count = int(condition_id.size)
        if count != 25 or np.unique(condition_id).size != count:
            raise ValueError("Replay must contain 24 corrupted conditions and one identity")
        family = _vector(item["family"], "family", length=count).astype(str)
        severity = _vector(item["severity"], "severity", length=count).astype(np.float32)
        replicate = _vector(item["replicate"], "replicate", length=count).astype(np.int32)
        identity = _vector(item["identity"], "identity", length=count).astype(bool)
        center = _vector(item["center"], "center", length=count).astype(np.float32)
        scale = _vector(item["scale"], "scale", length=count).astype(np.float32)
        if int(identity.sum()) != 1 or not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("Replay identity/normalization arrays are invalid")
        if np.any(scale <= 0):
            raise ValueError("Replay scales must be positive")
        series: dict[str, NDArray[np.float32]] = {}
        for key in SERIES_KEYS:
            values = np.asarray(item[key], dtype=np.float32)
            if values.shape != (count, reference.size) or not np.isfinite(values).all():
                raise ValueError(f"Replay series {key!r} has an invalid shape or value")
            series[key] = np.ascontiguousarray(values)

    replay = BenchmarkReplayCase(
        path=path,
        metadata=metadata,
        reference=np.ascontiguousarray(reference),
        labels=np.ascontiguousarray(labels),
        condition_id=condition_id.astype(str),
        family=family,
        severity=severity,
        replicate=replicate,
        identity=identity,
        center=center,
        scale=scale,
        series=series,
    )
    replay.default_index  # validate the declared default before returning
    return replay


def os_nrmse(
    reference: NDArray[np.float32],
    output: NDArray[np.float32],
    scale: float,
    labels: NDArray[np.bool_] | None = None,
) -> float | None:
    """Condition-level observation-scale NRMSE used by the benchmark."""

    mask = np.ones(reference.size, dtype=bool) if labels is None else np.asarray(labels, dtype=bool)
    if not np.any(mask):
        return None
    error = np.asarray(output, dtype=np.float64)[mask] - np.asarray(reference, dtype=np.float64)[mask]
    return float(np.sqrt(np.mean(error * error)) / float(scale))


class BenchmarkReplaySession:
    """Reversible observation-weight edits around the frozen model output."""

    def __init__(self, observation: NDArray[np.float32], baseline: NDArray[np.float32]) -> None:
        self.observation = np.ascontiguousarray(observation, dtype=np.float32)
        self.baseline = np.ascontiguousarray(baseline, dtype=np.float32)
        if self.observation.shape != self.baseline.shape:
            raise ValueError("Replay observation and baseline must have equal shape")
        self.current = self.baseline.copy()
        self._history: list[NDArray[np.float32]] = []
        self.revision = 0
        self.actions: list[dict[str, Any]] = []

    @property
    def history_depth(self) -> int:
        return len(self._history)

    def apply(
        self,
        action: str,
        *,
        start: int = 0,
        end: int | None = None,
        beta: float = 0.5,
        expected_revision: int | None = None,
    ) -> None:
        if expected_revision is not None and int(expected_revision) != self.revision:
            raise RuntimeError(
                f"Stale intervention revision: expected {expected_revision}, current {self.revision}."
            )
        if action == "revert":
            if not self._history:
                raise RuntimeError("No expert adaptation is available to revert.")
            self.current = self._history.pop()
            self._record(action, 0, self.current.size, None)
            return
        self._history.append(self.current.copy())
        if action == "restore_automatic":
            self.current = self.baseline.copy()
            self._record(action, 0, self.current.size, None)
            return
        stop = self.current.size if end is None else int(end)
        start = int(start)
        if not 0 <= start < stop <= self.current.size:
            self._history.pop()
            raise ValueError("Expert interval must satisfy 0 <= start < end <= 512")
        if action != "blend" or not np.isfinite(beta) or not 0.0 <= float(beta) <= 1.0:
            self._history.pop()
            raise ValueError("Replay supports a blend with observation weight in [0, 1]")
        # Expert edits accumulate across disjoint intervals.  The selected
        # interval is always recomputed from the immutable automatic baseline,
        # while earlier edits outside it remain intact and reversible.
        adapted = self.current.copy()
        adapted[start:stop] = (
            float(beta) * self.observation[start:stop]
            + (1.0 - float(beta)) * self.baseline[start:stop]
        )
        self.current = np.ascontiguousarray(adapted, dtype=np.float32)
        self._record(action, start, stop, float(beta))

    def _record(self, action: str, start: int, end: int, beta: float | None) -> None:
        self.revision += 1
        self.actions.append(
            {
                "revision": self.revision,
                "action": action,
                "start": int(start),
                "end": int(end),
                "beta": beta,
            }
        )
