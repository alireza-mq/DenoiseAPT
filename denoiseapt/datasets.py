"""Dataset loading, deterministic corruptions, and TSB-AD-U acquisition.

The demo works without network access through :func:`builtin_demo_case`.  The
download helper is deliberately dependency-free and performs guarded ZIP
extraction; it never calls ``ZipFile.extractall``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import ssl
import stat
import tempfile
from typing import Any, Iterable, Iterator, Sequence
from urllib.request import Request, urlopen
import warnings
import zipfile
import zlib

import numpy as np


TSB_AD_U_URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
# Pinned official archive observed 2026-08-13 (HTTP Last-Modified: 2026-05-10).
TSB_AD_U_SHA256 = "0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961"
TSB_AD_PROJECT_URL = "https://github.com/TheDatumOrg/TSB-AD"
TSB_AD_LICENSES_URL = "https://thedatumorg.github.io/TSB-AD/#summary-of-datasets"
GUIDED_TSB_AD_MEMBER = "TSB-AD-U/442_UCR_id_140_Medical_tr_1875_1st_4187.csv"

_LABEL_NAMES = ("label", "labels", "anomaly", "is_anomaly", "ground_truth", "gt")
_TIME_NAMES = ("time", "timestamp", "datetime", "date", "index")
_SIGNAL_NAMES = ("value", "signal", "measurement", "metric", "data")


@dataclass(frozen=True)
class TimeSeriesCase:
    """A validated univariate time series and optional point labels."""

    name: str
    time: np.ndarray
    signal: np.ndarray
    labels: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.time)
        signal = np.asarray(self.signal, dtype=np.float64)
        raw_labels = np.asarray(self.labels)
        if np.issubdtype(raw_labels.dtype, np.number) and not np.all(
            np.isfinite(raw_labels.astype(np.float64))
        ):
            raise ValueError("labels contain NaN or infinite values")
        labels = raw_labels.astype(bool)
        if signal.ndim != 1 or time.ndim != 1 or labels.ndim != 1:
            raise ValueError("time, signal, and labels must be one-dimensional")
        if not (len(time) == len(signal) == len(labels)):
            raise ValueError("time, signal, and labels must have equal lengths")
        if len(signal) < 2:
            raise ValueError("a time-series case needs at least two samples")
        if not np.all(np.isfinite(signal)):
            raise ValueError("signal contains NaN or infinite values")
        object.__setattr__(self, "time", time.copy())
        object.__setattr__(self, "signal", signal.copy())
        object.__setattr__(self, "labels", labels.copy())
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for the demo API."""

        return {
            "name": self.name,
            "time": [_json_scalar(value) for value in self.time],
            "signal": self.signal.tolist(),
            "labels": self.labels.astype(int).tolist(),
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class PreparedCase:
    """Reference and deterministically corrupted observation used by the demo."""

    source: TimeSeriesCase
    observation: np.ndarray
    corruption: str
    severity: float
    seed: int

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float64)
        if observation.shape != self.source.signal.shape:
            raise ValueError("observation shape must match the reference signal")
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation contains NaN or infinite values")
        object.__setattr__(self, "observation", observation.copy())

    def to_dict(self) -> dict[str, Any]:
        payload = self.source.to_dict()
        payload.update(
            {
                "reference": payload.pop("signal"),
                "observation": self.observation.tolist(),
                "corruption": self.corruption,
                "severity": self.severity,
                "seed": self.seed,
            }
        )
        return payload


def builtin_demo_case(length: int = 512, seed: int = 2026) -> TimeSeriesCase:
    """Create the deterministic, redistributable offline smoke-test fixture.

    The fixture represents a slowly varying sensor waveform with a short,
    low-amplitude anomalous event.  It is synthetic and is not a TSB-AD sample.
    """

    if length < 128:
        raise ValueError("length must be at least 128")
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float64)
    phase = np.linspace(0.0, 12.0 * np.pi, length, endpoint=False)
    normal = 0.72 * np.sin(phase) + 0.18 * np.sin(0.23 * phase + 0.7)
    normal += 0.0007 * time
    normal += rng.normal(0.0, 0.012, length)

    start = int(round(length * 0.46))
    width = max(10, int(round(length * 0.045)))
    stop = min(length, start + width)
    event_phase = np.linspace(0.0, np.pi, stop - start)
    event = 0.38 * np.sin(event_phase) ** 2
    signal = normal.copy()
    signal[start:stop] += event
    labels = np.zeros(length, dtype=bool)
    labels[start:stop] = True
    return TimeSeriesCase(
        name="synthetic_low_amplitude_event",
        time=time,
        signal=signal,
        labels=labels,
        metadata={
            "source": "DenoiseAPT deterministic synthetic fixture",
            "license": "Generated with this package; redistributable with the package license",
            "synthetic": True,
            "generator_seed": seed,
            "anomaly_interval": [start, stop],
        },
    )


def write_case_csv(case: TimeSeriesCase, path: str | Path) -> Path:
    """Write a case using the portable ``time,value,label`` schema."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "value", "label"])
        for t, value, label in zip(case.time, case.signal, case.labels):
            writer.writerow([_json_scalar(t), f"{value:.17g}", int(label)])
    return destination


def write_case_npz(
    case: TimeSeriesCase,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the safe, pickle-free NPZ schema consumed by the demo server."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined_metadata = dict(case.metadata)
    if metadata:
        combined_metadata.update(metadata)
    combined_metadata.setdefault("name", case.name)
    np.savez_compressed(
        destination,
        signal=case.signal.astype(np.float64),
        labels=case.labels.astype(np.int8),
        metadata_json=np.asarray(json.dumps(_json_safe(combined_metadata), sort_keys=True)),
    )
    return destination


def load_csv(
    path: str | Path,
    *,
    signal_column: str | None = None,
    label_column: str | None = None,
    time_column: str | None = None,
    max_rows: int | None = None,
) -> TimeSeriesCase:
    """Load a univariate CSV, including the standard TSB-AD ``value,Label`` form.

    Column matching is case-insensitive.  If no conventional signal header is
    present, the first numeric non-time, non-label column is used.  Rows with a
    missing signal are skipped; malformed non-empty values produce an error.
    """

    source = Path(path)
    if max_rows is not None and max_rows < 2:
        raise ValueError("max_rows must be at least two")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        lowered = {name.casefold(): name for name in fieldnames}
        label_name = _resolve_column(label_column, lowered, _LABEL_NAMES)
        time_name = _resolve_column(time_column, lowered, _TIME_NAMES)
        signal_name = _resolve_signal_column(
            signal_column, lowered, fieldnames, label_name, time_name
        )

        times: list[Any] = []
        values: list[float] = []
        labels: list[bool] = []
        for row_number, raw_row in enumerate(reader, start=2):
            row = {(key or "").strip(): value for key, value in raw_row.items()}
            raw_signal = (row.get(signal_name) or "").strip()
            if not raw_signal:
                continue
            try:
                value = float(raw_signal)
            except ValueError as exc:
                raise ValueError(
                    f"non-numeric signal at CSV row {row_number}: {raw_signal!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite signal at CSV row {row_number}")
            values.append(value)
            times.append(
                _parse_time((row.get(time_name) or "").strip(), len(values) - 1)
                if time_name
                else len(values) - 1
            )
            labels.append(
                _parse_label((row.get(label_name) or "0").strip(), row_number)
                if label_name
                else False
            )
            if max_rows is not None and len(values) >= max_rows:
                break

    return TimeSeriesCase(
        name=source.stem,
        time=np.asarray(times),
        signal=np.asarray(values, dtype=np.float64),
        labels=np.asarray(labels, dtype=bool),
        metadata={
            "source_file": str(source.resolve()),
            "signal_column": signal_name,
            "label_column": label_name,
            "time_column": time_name,
        },
    )


def apply_corruption(
    signal: Sequence[float] | np.ndarray,
    kind: str = "gaussian",
    severity: float = 0.25,
    seed: int = 7,
) -> np.ndarray:
    """Apply the same deterministic corruption implementation used by the API."""

    from .corruptions import apply_measurement_corruption

    normalized_kind = kind.strip().casefold()
    return np.asarray(
        apply_measurement_corruption(
            signal,
            kind=normalized_kind,
            severity=severity,
            seed=seed,
        ).corrupted,
        dtype=np.float64,
    )


def extract_window(
    case: TimeSeriesCase, length: int, start: int | None = None
) -> TimeSeriesCase:
    """Extract a fixed window, centring the first labelled event by default."""

    if length < 2:
        raise ValueError("window length must be at least two")
    if length >= len(case.signal):
        return case
    automatic_selection = start is None
    if automatic_selection:
        anomaly_points = np.flatnonzero(case.labels)
        centre = (
            int(np.median(anomaly_points))
            if len(anomaly_points)
            else len(case.signal) // 2
        )
        start = centre - length // 2
    start = min(max(int(start), 0), len(case.signal) - length)
    stop = start + length
    metadata = dict(case.metadata)
    metadata.update(
        {
            "source_window": [start, stop],
            "source_index_start": start,
            "source_index_stop": stop,
            "window_selection": (
                "centred on first labelled event"
                if automatic_selection
                else "explicit start"
            ),
        }
    )
    return TimeSeriesCase(
        name=f"{case.name}_{start}_{stop}",
        time=case.time[start:stop],
        signal=case.signal[start:stop],
        labels=case.labels[start:stop],
        metadata=metadata,
    )


def prepare_case(
    case: TimeSeriesCase,
    *,
    corruption: str = "mixed",
    severity: float = 0.25,
    seed: int = 7,
    window_length: int | None = 512,
    start: int | None = None,
) -> PreparedCase:
    """Window a case and produce the controlled observation used by the UI."""

    selected = extract_window(case, window_length, start) if window_length else case
    observation = apply_corruption(selected.signal, corruption, severity, seed)
    return PreparedCase(
        selected,
        observation,
        corruption.strip().casefold(),
        float(severity),
        int(seed),
    )


def discover_tsb_csv(root: str | Path) -> list[Path]:
    """Return TSB-AD CSV files in deterministic order."""

    directory = Path(root)
    return sorted(path for path in directory.rglob("*.csv") if path.is_file())


def iter_training_cases(
    data_dir: str | Path,
    *,
    max_cases: int | None = None,
    min_length: int = 128,
) -> Iterator[TimeSeriesCase]:
    """Yield validated TSB-AD CSV cases in stable path order.

    Signals are returned in their original scale; the model training pipeline
    remains responsible for fitting and recording any normalization.
    """

    if min_length < 2:
        raise ValueError("min_length must be at least two")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    yielded = 0
    for path in discover_tsb_csv(data_dir):
        case = load_csv(path)
        if len(case.signal) < min_length:
            continue
        yield case
        yielded += 1
        if max_cases is not None and yielded >= max_cases:
            return


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secure_extract_zip(
    archive: str | Path,
    destination: str | Path,
    *,
    members: Iterable[str] | None = None,
    max_members: int = 20_000,
    max_total_bytes: int = 20 * 1024**3,
    overwrite: bool = False,
) -> list[Path]:
    """Extract selected ZIP members after traversal, symlink, and size checks."""

    archive_path = Path(archive)
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    root = target_root.resolve()
    selected_names = set(members) if members is not None else None

    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as bundle:
        infos = [
            info
            for info in bundle.infolist()
            if selected_names is None or info.filename in selected_names
        ]
        if len(infos) > max_members:
            raise ValueError(
                "archive contains too many selected members "
                f"({len(infos)} > {max_members})"
            )
        total = sum(info.file_size for info in infos if not info.is_dir())
        if total > max_total_bytes:
            raise ValueError(f"selected archive content is too large ({total} bytes)")

        # Validate every destination before writing the first archive member.
        validated: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            relative = _safe_zip_member_path(info)
            output = (root / Path(*relative.parts)).resolve()
            if root != output and root not in output.parents:
                raise ValueError(f"unsafe ZIP path: {info.filename!r}")
            validated.append((info, output))

        for info, output in validated:
            if info.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists() and not overwrite:
                if output.stat().st_size == info.file_size and _crc32_file(output) == info.CRC:
                    extracted.append(output)
                    continue
                raise FileExistsError(f"refusing to overwrite existing file: {output}")
            with bundle.open(info) as source, output.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            extracted.append(output)
    return extracted


def download_tsb_ad_u(
    destination: str | Path,
    *,
    cache_dir: str | Path | None = None,
    subset: Sequence[str] | None = None,
    max_files: int | None = None,
    url: str = TSB_AD_U_URL,
    expected_sha256: str | None = TSB_AD_U_SHA256,
    force_download: bool = False,
    timeout: float = 60.0,
    max_download_bytes: int = 8 * 1024**3,
    allow_insecure: bool = False,
) -> dict[str, Any]:
    """Download and securely extract TSB-AD-U, optionally selecting CSVs.

    ``subset`` contains case-insensitive filename/path fragments such as
    ``["NAB", "YAHOO"]``.  ``max_files`` bounds the number of extracted CSVs.
    The returned dictionary is also saved as ``PROVENANCE.json``.
    """

    destination_path = Path(destination).expanduser().resolve()
    cache_path = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else destination_path.parent / "cache"
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    archive = cache_path / "TSB-AD-U.zip"
    download_performed = force_download or not archive.exists()
    if allow_insecure and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256.strip()) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        raise ValueError(
            "allow_insecure requires a non-empty 64-character expected_sha256 "
            "from a trusted channel"
        )
    if download_performed:
        _download_file(
            url,
            archive,
            timeout=timeout,
            max_bytes=max_download_bytes,
            allow_insecure=allow_insecure,
        )
    digest = sha256_file(archive)
    if expected_sha256 and digest.casefold() != expected_sha256.casefold():
        raise ValueError(
            f"SHA-256 mismatch for {archive}: "
            f"expected {expected_sha256}, got {digest}"
        )

    acquisition_path = cache_path / "TSB-AD-U.zip.provenance.json"
    acquisition: dict[str, Any] = {}
    if download_performed:
        acquisition = {
            "download_url": url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": digest,
            "tls_verification_disabled": bool(allow_insecure),
        }
        acquisition_path.write_text(json.dumps(acquisition, indent=2) + "\n", encoding="utf-8")
    elif acquisition_path.is_file():
        try:
            acquisition = json.loads(acquisition_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            acquisition = {"status": "cached archive acquisition metadata unreadable"}

    patterns = [item.casefold() for item in (subset or []) if item.strip()]
    with zipfile.ZipFile(archive) as bundle:
        candidates = sorted(
            info.filename
            for info in bundle.infolist()
            if not info.is_dir()
            and info.filename.casefold().endswith(".csv")
            and (not patterns or any(pattern in info.filename.casefold() for pattern in patterns))
        )
    if not candidates:
        raise ValueError(f"no CSV members matched subset {list(subset or [])!r}")
    if max_files is not None:
        if max_files < 1:
            raise ValueError("max_files must be positive")
        candidates = candidates[:max_files]

    files = secure_extract_zip(archive, destination_path, members=candidates)
    provenance = {
        "dataset": "TSB-AD-U",
        "download_url": url,
        "project_url": TSB_AD_PROJECT_URL,
        "per_dataset_license_url": TSB_AD_LICENSES_URL,
        "provenance_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned_archive_last_modified": "2026-05-10",
        "archive_sha256": digest,
        "archive_cache": str(archive.resolve()),
        "download_performed": download_performed,
        "tls_verification_disabled_for_download": bool(
            allow_insecure and download_performed
        ),
        "archive_acquisition": acquisition or {"status": "unknown for pre-existing cache"},
        "selection_fragments": list(subset or []),
        "selected_member_count": len(files),
        "selected_members": [path.relative_to(destination_path).as_posix() for path in files],
        "license_notice": (
            "TSB-AD preprocessing and curation code is Apache-2.0; contained datasets retain "
            "their original licenses. Consult per_dataset_license_url and cite "
            "the upstream source."
        ),
    }
    destination_path.mkdir(parents=True, exist_ok=True)
    with (destination_path / "PROVENANCE.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    return provenance


def _download_file(
    url: str,
    destination: Path,
    *,
    timeout: float,
    max_bytes: int,
    allow_insecure: bool,
) -> None:
    request = Request(
        url,
        headers={"User-Agent": "DenoiseAPT-Demo/0.2.2 (+research artifact)"},
    )
    context = None
    if allow_insecure:
        warnings.warn(
            "TLS certificate verification is DISABLED for this dataset download; "
            "verify the archive SHA-256 through a trusted channel.",
            RuntimeWarning,
            stacklevel=2,
        )
        # Explicit, opt-in CLI escape hatch; the archive hash remains mandatory.
        context = ssl._create_unverified_context()  # noqa: SLF001
    temporary: Path | None = None
    try:
        with urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310
            length_header = response.headers.get("Content-Length")
            if length_header and int(length_header) > max_bytes:
                raise ValueError(
                    "download is larger than the configured limit "
                    f"({max_bytes} bytes)"
                )
            with tempfile.NamedTemporaryFile(
                "wb", delete=False, dir=destination.parent, suffix=".part"
            ) as handle:
                temporary = Path(handle.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            "download exceeded the configured limit "
                            f"({max_bytes} bytes)"
                        )
                    handle.write(chunk)
        if temporary is None:
            raise RuntimeError("download produced no temporary file")
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_zip_member_path(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    unix_mode = info.external_attr >> 16
    if stat.S_ISLNK(unix_mode):
        raise ValueError(f"symbolic links are not allowed in ZIP archives: {info.filename!r}")
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe ZIP path: {info.filename!r}")
    if ":" in path.parts[0]:
        raise ValueError(f"drive-qualified ZIP path is not allowed: {info.filename!r}")
    return path


def _resolve_column(
    requested: str | None,
    lowered: dict[str, str],
    conventional: Sequence[str],
) -> str | None:
    if requested is not None:
        resolved = lowered.get(requested.strip().casefold())
        if resolved is None:
            raise ValueError(f"CSV column not found: {requested!r}")
        return resolved
    return next((lowered[name] for name in conventional if name in lowered), None)


def _resolve_signal_column(
    requested: str | None,
    lowered: dict[str, str],
    fieldnames: list[str],
    label_name: str | None,
    time_name: str | None,
) -> str:
    if requested is not None:
        resolved = lowered.get(requested.strip().casefold())
        if resolved is None:
            raise ValueError(f"CSV column not found: {requested!r}")
        return resolved
    conventional = next((lowered[name] for name in _SIGNAL_NAMES if name in lowered), None)
    if conventional:
        return conventional
    candidates = [name for name in fieldnames if name not in {label_name, time_name}]
    if not candidates:
        raise ValueError("CSV has no candidate signal column")
    return candidates[0]


def _parse_label(raw: str, row_number: int) -> bool:
    normalized = raw.casefold()
    if normalized in {"", "0", "false", "normal", "no"}:
        return False
    if normalized in {"1", "true", "anomaly", "anomalous", "yes"}:
        return True
    try:
        return float(raw) != 0.0
    except ValueError as exc:
        raise ValueError(f"invalid label at CSV row {row_number}: {raw!r}") from exc


def _parse_time(raw: str, fallback: int) -> Any:
    if not raw:
        return fallback
    try:
        value = float(raw)
        return int(value) if value.is_integer() else value
    except ValueError:
        return raw


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _json_scalar(value)


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF
