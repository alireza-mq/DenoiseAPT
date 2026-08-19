"""Retrospective matched benchmark against runnable time-series denoisers.

This additive runner deliberately does not alter the completed Hybrid-v2
confirmation.  Hyperparameters for Wavelet Shrinkage and Noisereduce are
chosen using development validation data only.  A pre-evaluation receipt then
binds the selection, code, packages, adapted RINS-T source, and already-spent
audit traces before any new comparator output is generated on that panel.

The result is a descriptive matched-data benchmark, not a new confirmation
experiment and not evidence of universal state of the art.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).absolute().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = Path(__file__).with_name("external_denoiser_benchmark_v1.json")
DEPENDENCY_MANIFEST_PATH = Path(__file__).with_name(
    "external_denoiser_benchmark_v1_dependencies.json"
)
THIRD_PARTY_NOTICE_PATH = (
    PROJECT_ROOT / "third_party" / "benchmarks" / "THIRD_PARTY_NOTICES.md"
)
OUTPUT_PATH = PROJECT_ROOT / "runs" / "external_denoiser_benchmark_v1"
SELECTION_PATH = OUTPUT_PATH / "development_selection.json"
PREFREEZE_PATH = OUTPUT_PATH / "PREFREEZE.json"
ARTIFACT_DIR = OUTPUT_PATH / "artifacts"
SUMMARY_PATH = ARTIFACT_DIR / "benchmark_summary.json"
OUTPUTS_PATH = ARTIFACT_DIR / "benchmark_outputs.npz"
WAVEFORM_ROWS_PATH = ARTIFACT_DIR / "waveform_rows.csv"
WITNESS_ROWS_PATH = ARTIFACT_DIR / "witness_rows.csv"
RUNNER_PATH = Path(__file__).absolute()
DEVELOPMENT_PATH = PROJECT_ROOT / "data" / "experiment" / "quality_v4_development_v2.npz"
TRACE_DIR = PROJECT_ROOT / "runs" / "hybrid_v2_confirmation_fresh_v1" / "units" / "seed_17"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "runs"
    / "hybrid_v2_confirmation_fresh_v1"
    / "artifacts"
    / "calibration_thresholds.json"
)
RINS_DIR = PROJECT_ROOT / "third_party" / "benchmarks" / "RINS-T"
RUNTIME_MANIFEST = PROJECT_ROOT / "checkpoints" / "automatic_preservation" / "runtime_manifest.json"

DOMAINS = ("sensor", "medical")
METHODS = (
    "corrupted_input",
    "median_filter_w3",
    "wavelet_shrinkage",
    "noisereduce",
    "rins_t",
    "denoiseapt",
)
CONFIGURED_SCORERS = ("A_causal_mlp", "B_causal_conv")
FAMILIES = ("gaussian", "impulse", "drift", "mixed")
SEVERITIES = (0.25, 0.50, 0.75)


class BenchmarkError(RuntimeError):
    """Raised when an audit or execution contract differs."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BenchmarkError(f"JSON root must be an object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    """Hash import-relevant package bytes without caches or metadata timestamps."""

    digest = hashlib.sha256()
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and "__pycache__" not in item.parts
        and item.suffix.lower() not in {".pyc", ".pyo"}
    ]
    for item in sorted(files, key=lambda value: value.relative_to(path).as_posix()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _package_identity() -> dict[str, Any]:
    import noisereduce
    import pywt
    import scipy

    noisereduce_root = Path(noisereduce.__file__).resolve().parent
    pywt_root = Path(pywt.__file__).resolve().parent
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "noisereduce_distribution": importlib.metadata.version("noisereduce"),
        "noisereduce_tree_sha256": _tree_sha256(noisereduce_root),
        "pywavelets_distribution": importlib.metadata.version("PyWavelets"),
        "pywt_runtime": str(pywt.__version__),
        "pywt_tree_sha256": _tree_sha256(pywt_root),
    }


def _validate_dependency_manifest() -> dict[str, Any]:
    import torch

    manifest = _load_json(DEPENDENCY_MANIFEST_PATH)
    runtime = manifest.get("runtime", {})
    methods = manifest.get("external_methods", {})
    wavelet = methods.get("wavelet_shrinkage", {})
    spectral = methods.get("noisereduce", {})
    rins = methods.get("rins_t", {})
    package = _package_identity()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_id")
        != "denoiseapt-external-denoiser-benchmark-retrospective-v1"
        or runtime
        != {
            "python": "3.12.13",
            "numpy": "2.5.2",
            "scipy": "1.18.0",
            "torch": "2.13.0+cpu",
        }
        or package["python"] != runtime["python"]
        or package["numpy"] != runtime["numpy"]
        or package["scipy"] != runtime["scipy"]
        or torch.__version__ != runtime["torch"]
        or wavelet.get("package") != "PyWavelets"
        or wavelet.get("version") != package["pywavelets_distribution"]
        or wavelet.get("repository") != "https://github.com/PyWavelets/pywt"
        or wavelet.get("license") != "MIT"
        or wavelet.get("foundation_doi") != "10.1109/18.382009"
        or spectral
        != {
            "package": "noisereduce",
            "version": "3.0.3",
            "repository": "https://github.com/timsainb/noisereduce",
            "license": "MIT",
            "paper_doi": "10.1038/s41598-025-13108-x",
        }
        or rins
        != {
            "repository": "https://github.com/EPFL-IMOS/RINS-T",
            "commit": "95d1d9b44b44ba771b2400f5fb68fe42447c5fa2",
            "license": None,
            "license_status": "No software license was found in the repository at the pinned commit.",
            "paper_doi": "10.1109/TIM.2025.3632427",
            "implementation_role": "Official architecture and demo recipe adapted to the benchmark's observation-only input wrapper and fixed optimization loop.",
        }
    ):
        raise BenchmarkError("benchmark dependency manifest semantics differ")
    return manifest


def _rins_commit() -> str:
    git = RINS_DIR / ".git"
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        commit = (git / head[5:]).read_text(encoding="utf-8").strip()
    else:
        commit = head
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise BenchmarkError("RINS-T repository HEAD is invalid")
    return commit


def _tuning_dependency_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "denoiseapt" / "corruptions.py",
        PROJECT_ROOT / "data" / "experiment" / "quality_v4_development_v2_manifest.json",
        PROJECT_ROOT / "data" / "experiment" / "quality_v4_development_v2_LOCK.json",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256_file(path)
        for path in paths
    }


def _array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise BenchmarkError(f"exclusive output already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise BenchmarkError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise BenchmarkError(f"CSV row schema differs: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _validate_config(config: Mapping[str, Any]) -> None:
    scope = config.get("scientific_scope", {})
    development = config.get("development", {})
    wavelet = config.get("wavelet_grid", {})
    spectral = config.get("noisereduce_grid", {})
    rins = config.get("rins_t", {})
    evaluation = config.get("evaluation", {})
    sources = config.get("method_sources", {})
    if (
        config.get("schema_version") != 1
        or config.get("protocol_id")
        != "denoiseapt-external-denoiser-benchmark-retrospective-v1"
        or config.get("status") != "DEVELOPMENT_TUNING_ONLY_BEFORE_EXTERNAL_TEST_OUTPUTS"
        or tuple(config.get("methods", ())) != METHODS
        or scope
        != {
            "retrospective": True,
            "panel_opened_before_comparator_roster_was_selected": True,
            "confirmation_claim_eligible": False,
            "state_of_the_art_claim_eligible": False,
            "published_scores_copied_into_table": False,
            "comparisons_are_descriptive": True,
        }
        or sources
        != {
            "wavelet_shrinkage": {
                "foundation_doi": "10.1109/18.382009",
                "foundation_scope": "foundational soft-thresholding paper; the benchmark selects soft or hard thresholding on development data",
                "software_repository": "https://github.com/PyWavelets/pywt",
                "software_license": "MIT",
                "software_version": "1.8.0",
            },
            "noisereduce": {
                "paper_doi": "10.1038/s41598-025-13108-x",
                "software_repository": "https://github.com/timsainb/noisereduce",
                "software_license": "MIT",
                "software_version": "3.0.3",
            },
            "rins_t": {
                "paper_doi": "10.1109/TIM.2025.3632427",
                "software_repository": "https://github.com/EPFL-IMOS/RINS-T",
                "software_license": None,
                "license_status": "no software license found at the pinned commit",
            },
        }
        or development.get("path") != "data/experiment/quality_v4_development_v2.npz"
        or development.get("partition") != "validation"
        or development.get("window_selection") != "minimum_sha256_per_source_group"
        or development.get("selection_namespace")
        != "external-denoiser-benchmark-validation-window-v1"
        or development.get("corruption_namespace")
        != "external-denoiser-benchmark-validation-corruption-v1"
        or tuple(development.get("families", ())) != FAMILIES
        or tuple(map(float, development.get("severities", ()))) != SEVERITIES
        or tuple(map(int, development.get("replicates", ()))) != (0,)
        or development.get("labels_used_for_selection") is not False
        or wavelet.get("package") != "PyWavelets"
        or wavelet.get("package_version") != "1.8.0"
        or tuple(wavelet.get("wavelets", ())) != ("db2", "db4", "sym4")
        or tuple(wavelet.get("threshold_modes", ())) != ("soft", "hard")
        or tuple(map(float, wavelet.get("threshold_multipliers", ()))) != (0.5, 1.0, 1.5)
        or int(wavelet.get("level", -1)) != 4
        or wavelet.get("boundary_mode") != "symmetric"
        or wavelet.get("noise_scale")
        != "MAD of finest detail divided by 0.6744897501960817"
        or wavelet.get("base_threshold") != "sigma*sqrt(2*log(512))"
        or spectral.get("package_version") != "3.0.3"
        or spectral.get("input_preprocessing")
        != "per-window median/robust-scale normalization, restored to original units"
        or tuple(spectral.get("stationary", ())) != (True, False)
        or tuple(map(float, spectral.get("prop_decrease", ()))) != (0.5, 0.75, 1.0)
        or tuple(map(int, spectral.get("n_fft", ()))) != (32, 64)
        or int(spectral.get("normalized_sample_rate", -1)) != 512
        or spectral.get("sample_rate_semantics")
        != "algorithmic index rate for a normalized 512-sample window; not a measured physical sampling frequency"
        or int(spectral.get("nonstationary_time_constant_samples", -1)) != 64
        or spectral.get("frequency_mask_smoothing") is not None
        or spectral.get("time_mask_smoothing") is not None
        or rins.get("repository_commit") != "95d1d9b44b44ba771b2400f5fb68fe42447c5fa2"
        or _rins_commit() != rins.get("repository_commit")
        or rins.get("software_license_present") is not False
        or rins.get("redistribution_allowed_by_this_benchmark") is not False
        or rins.get("normalization") != "per-window min-max to [0,1]"
        or rins.get("guided_input") != "Gaussian filter sigma 4 samples"
        or int(rins.get("iterations", -1)) != 27
        or rins.get("optimizer") != "Adam"
        or float(rins.get("learning_rate", -1)) != 0.01
        or rins.get("loss") != "Huber"
        or float(rins.get("huber_delta", -1)) != 0.001
        or float(rins.get("input_noise_std", -1)) != 0.03
        or float(rins.get("output_ema_alpha", -1)) != 0.5
        or int(rins.get("seed_per_condition", -1)) != 42
        or int(rins.get("workers", -1)) != 4
        or rins.get("official_repository") != "https://github.com/EPFL-IMOS/RINS-T"
        or rins.get("implementation_role")
        != "internal official-architecture/demo-recipe adaptation"
        or evaluation.get("source")
        != "runs/hybrid_v2_confirmation_fresh_v1/units/seed_17/{domain}/audit_trace.npz"
        or tuple(evaluation.get("domains", ())) != DOMAINS
        or int(evaluation.get("conditions", -1)) != 2700
        or int(evaluation.get("corrupted_conditions", -1)) != 2592
        or int(evaluation.get("source_groups", -1)) != 32
        or int(evaluation.get("windows", -1)) != 108
        or tuple(evaluation.get("configured_scorers", ())) != CONFIGURED_SCORERS
        or evaluation.get("thresholds")
        != "runs/hybrid_v2_confirmation_fresh_v1/artifacts/calibration_thresholds.json"
        or evaluation.get("denoiseapt_trace_method") != "output__hybrid_v2"
        or config.get("output") != "runs/external_denoiser_benchmark_v1"
    ):
        raise BenchmarkError("benchmark configuration semantics differ")
    package = _package_identity()
    if (
        package["pywavelets_distribution"] != wavelet["package_version"]
        or package["pywt_runtime"] != wavelet["package_version"]
        or package["noisereduce_distribution"] != spectral["package_version"]
    ):
        raise BenchmarkError("benchmark package versions differ from configuration")
    _validate_dependency_manifest()


def _validate_selection(selection: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    value = dict(selection)
    expected_sha = value.pop("selection_sha256", None)
    if not isinstance(expected_sha, str) or _canonical_sha256(value) != expected_sha:
        raise BenchmarkError("development selection self-hash differs")
    wavelet_rows = selection.get("wavelet_grid", [])
    noisereduce_rows = selection.get("noisereduce_grid", [])
    if (
        selection.get("schema_version") != 1
        or selection.get("protocol_id") != config.get("protocol_id")
        or selection.get("status")
        != "DEVELOPMENT_SELECTION_COMPLETE_TEST_OUTPUTS_NOT_GENERATED"
        or selection.get("development_path_sha256") != _sha256_file(DEVELOPMENT_PATH)
        or selection.get("tuning_runner_sha256") != _sha256_file(RUNNER_PATH)
        or selection.get("tuning_config_sha256") != _sha256_file(CONFIG_PATH)
        or selection.get("tuning_dependency_sha256") != _tuning_dependency_hashes()
        or selection.get("package_identity") != _package_identity()
        or int(selection.get("selected_validation_windows", -1)) != 61
        or int(selection.get("validation_conditions", -1)) != 732
        or int(selection.get("source_groups", -1)) != 61
        or selection.get("labels_used") is not False
        or len(wavelet_rows) != 18
        or len(noisereduce_rows) != 12
    ):
        raise BenchmarkError("development selection semantics differ")
    expected_wavelet = min(
        wavelet_rows,
        key=lambda row: (
            float(row["validation_os_nrmse"]),
            str(row["wavelet"]),
            str(row["threshold_mode"]),
            float(row["threshold_multiplier"]),
        ),
    )
    expected_noisereduce = min(
        noisereduce_rows,
        key=lambda row: (
            float(row["validation_os_nrmse"]),
            int(not bool(row["stationary"])),
            float(row["prop_decrease"]),
            int(row["n_fft"]),
        ),
    )
    if (
        selection.get("selected_wavelet") != expected_wavelet
        or selection.get("selected_noisereduce") != expected_noisereduce
    ):
        raise BenchmarkError("development selection is not the deterministic grid argmin")


def observation_scale(values: NDArray[np.floating[Any]]) -> float:
    array = np.asarray(values, dtype=np.float64)
    q25, q75 = np.percentile(array, [25.0, 75.0])
    return max(float((q75 - q25) / 1.349), float(np.std(array)), 1e-4)


def median_filter_w3(values: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (512,):
        raise ValueError("median filter requires one 512-sample window")
    padded = np.pad(array, (1, 1), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 3)
    return np.asarray(np.median(windows, axis=-1), dtype=np.float32)


def wavelet_shrinkage(
    values: NDArray[np.floating[Any]],
    *,
    wavelet: str,
    threshold_mode: str,
    threshold_multiplier: float,
    level: int = 4,
) -> NDArray[np.float32]:
    import pywt

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (512,) or not np.all(np.isfinite(array)):
        raise ValueError("wavelet shrinkage requires one finite 512-sample window")
    coefficients = pywt.wavedec(array, wavelet, mode="symmetric", level=int(level))
    finest = np.asarray(coefficients[-1], dtype=np.float64)
    sigma = float(np.median(np.abs(finest - np.median(finest)))) / 0.6744897501960817
    threshold = float(threshold_multiplier) * sigma * math.sqrt(2.0 * math.log(len(array)))
    shrunk = [coefficients[0]] + [
        pywt.threshold(detail, threshold, mode=threshold_mode) for detail in coefficients[1:]
    ]
    restored = np.asarray(
        pywt.waverec(shrunk, wavelet, mode="symmetric")[: len(array)], dtype=np.float32
    )
    if restored.shape != (512,) or not np.all(np.isfinite(restored)):
        raise BenchmarkError("wavelet output is invalid")
    return restored


def noisereduce_filter(
    values: NDArray[np.floating[Any]],
    *,
    stationary: bool,
    prop_decrease: float,
    n_fft: int,
) -> NDArray[np.float32]:
    import noisereduce as nr

    array = np.asarray(values, dtype=np.float32)
    center = float(np.median(array))
    scale = observation_scale(array)
    normalized = np.asarray((array - center) / scale, dtype=np.float32)
    restored = nr.reduce_noise(
        y=normalized,
        sr=512,
        stationary=bool(stationary),
        prop_decrease=float(prop_decrease),
        time_constant_s=64.0 / 512.0,
        freq_mask_smooth_hz=None,
        time_mask_smooth_ms=None,
        n_fft=int(n_fft),
        win_length=int(n_fft),
        hop_length=int(n_fft) // 4,
        use_tqdm=False,
        n_jobs=1,
        use_torch=False,
    )
    output = np.asarray(np.asarray(restored, dtype=np.float32) * scale + center, dtype=np.float32)
    if output.shape != (512,) or not np.all(np.isfinite(output)):
        raise BenchmarkError("Noisereduce output is invalid")
    return output


def _rins_t_worker(task: tuple[NDArray[np.float32], int]) -> NDArray[np.float32]:
    """Adapt the bound RINS-T architecture/demo recipe to one observation."""

    values, seed = task
    import torch
    from scipy.ndimage import gaussian_filter1d

    architecture_path = (RINS_DIR / "architecture.py").resolve()
    module_name = "denoiseapt_benchmark_rins_t_architecture_95d1d9b"
    specification = importlib.util.spec_from_file_location(module_name, architecture_path)
    if specification is None or specification.loader is None:
        raise BenchmarkError("cannot load the bound RINS-T architecture")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    if Path(module.__file__).resolve() != architecture_path:
        raise BenchmarkError("loaded RINS-T architecture path differs")
    skip = module.skip

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(seed))
    array = np.asarray(values, dtype=np.float32)
    low = float(np.min(array))
    high = float(np.max(array))
    span = high - low
    if not math.isfinite(span) or span <= 1e-12:
        return array.copy()
    scaled = np.asarray((array - low) / span, dtype=np.float32)
    target = torch.from_numpy(scaled)[None, None, :]
    guided = np.asarray(gaussian_filter1d(scaled, sigma=4.0, mode="reflect"), dtype=np.float32)
    net_input_saved = torch.from_numpy(guided)[None, None, :]
    network = skip().cpu().train()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    loss_function = torch.nn.HuberLoss(reduction="mean", delta=0.001)
    noise = torch.empty_like(net_input_saved)
    output_average: Any | None = None
    for _ in range(27):
        optimizer.zero_grad(set_to_none=True)
        model_input = net_input_saved + noise.normal_() * 0.03
        output = network(model_input)
        output_average = (
            output.detach()
            if output_average is None
            else output_average * 0.5 + output.detach() * 0.5
        )
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
    if output_average is None:
        raise BenchmarkError("RINS-T produced no optimization iterate")
    restored = np.asarray(output_average.squeeze().numpy() * span + low, dtype=np.float32)
    if restored.shape != (512,) or not np.all(np.isfinite(restored)):
        raise BenchmarkError("RINS-T output is invalid")
    return restored


def run_rins_t(windows: NDArray[np.float32], *, workers: int = 4) -> NDArray[np.float32]:
    values = np.asarray(windows, dtype=np.float32)
    tasks = [(row, 42) for row in values]
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        outputs = list(executor.map(_rins_t_worker, tasks, chunksize=4))
    result = np.stack(outputs).astype(np.float32)
    if result.shape != values.shape or not np.all(np.isfinite(result)):
        raise BenchmarkError("RINS-T output grid is invalid")
    return result


def _selected_validation_indices(archive: Mapping[str, NDArray[Any]]) -> list[int]:
    groups = np.asarray(archive["validation_source_group_id"]).astype(str)
    windows = np.asarray(archive["validation_window_id"]).astype(str)
    selected: list[int] = []
    for group in sorted(set(groups)):
        candidates = np.flatnonzero(groups == group)
        keyed = [
            (
                hashlib.sha256(
                    f"external-denoiser-benchmark-validation-window-v1|{group}|{windows[index]}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                int(index),
            )
            for index in candidates
        ]
        selected.append(min(keyed)[1])
    return selected


def _development_conditions() -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.str_]]:
    from denoiseapt.corruptions import apply_measurement_corruption

    with np.load(DEVELOPMENT_PATH, allow_pickle=False) as archive:
        indices = _selected_validation_indices(archive)
        clean = np.asarray(archive["validation_clean"], dtype=np.float32)
        groups = np.asarray(archive["validation_source_group_id"]).astype(str)
        windows = np.asarray(archive["validation_window_id"]).astype(str)
        references: list[NDArray[np.float32]] = []
        observations: list[NDArray[np.float32]] = []
        selected_groups: list[str] = []
        for index in indices:
            for family in FAMILIES:
                for severity in SEVERITIES:
                    seed = _stable_seed(
                        "external-denoiser-benchmark-validation-corruption-v1",
                        windows[index],
                        family,
                        severity,
                        0,
                    )
                    observation = apply_measurement_corruption(
                        clean[index], kind=family, severity=severity, seed=seed
                    ).corrupted
                    references.append(clean[index])
                    observations.append(observation)
                    selected_groups.append(groups[index])
    result = (
        np.ascontiguousarray(references, dtype=np.float32),
        np.ascontiguousarray(observations, dtype=np.float32),
        np.asarray(selected_groups).astype(str),
    )
    if result[0].shape != (61 * 12, 512) or len(set(result[2])) != 61:
        raise BenchmarkError("development selection cardinality differs")
    return result


def _group_balanced_os(
    reference: NDArray[np.float32],
    observation: NDArray[np.float32],
    output: NDArray[np.float32],
    groups: NDArray[np.str_],
) -> float:
    scales = np.asarray([observation_scale(row) for row in observation], dtype=np.float64)
    errors = np.sqrt(
        np.mean(np.square(output.astype(np.float64) - reference.astype(np.float64)), axis=1)
    ) / scales
    means = [float(np.mean(errors[groups == group])) for group in sorted(set(groups))]
    return float(np.mean(means))


def tune_development() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    _validate_config(config)
    if SELECTION_PATH.exists():
        return _load_json(SELECTION_PATH)
    if any(path.exists() for path in (PREFREEZE_PATH, SUMMARY_PATH, OUTPUTS_PATH)):
        raise BenchmarkError("cannot tune after evaluation lifecycle has advanced")
    reference, observation, groups = _development_conditions()
    wavelet_rows: list[dict[str, Any]] = []
    for wavelet in config["wavelet_grid"]["wavelets"]:
        for mode in config["wavelet_grid"]["threshold_modes"]:
            for multiplier in config["wavelet_grid"]["threshold_multipliers"]:
                began = time.perf_counter()
                output = np.stack(
                    [
                        wavelet_shrinkage(
                            row,
                            wavelet=str(wavelet),
                            threshold_mode=str(mode),
                            threshold_multiplier=float(multiplier),
                        )
                        for row in observation
                    ]
                )
                score = _group_balanced_os(reference, observation, output, groups)
                row = {
                    "config_id": f"wavelet_{wavelet}_{mode}_{float(multiplier):.2f}",
                    "wavelet": str(wavelet),
                    "threshold_mode": str(mode),
                    "threshold_multiplier": float(multiplier),
                    "validation_os_nrmse": score,
                    "elapsed_seconds": time.perf_counter() - began,
                }
                wavelet_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    noisereduce_rows: list[dict[str, Any]] = []
    for stationary in config["noisereduce_grid"]["stationary"]:
        for prop_decrease in config["noisereduce_grid"]["prop_decrease"]:
            for n_fft in config["noisereduce_grid"]["n_fft"]:
                began = time.perf_counter()
                output = np.stack(
                    [
                        noisereduce_filter(
                            row,
                            stationary=bool(stationary),
                            prop_decrease=float(prop_decrease),
                            n_fft=int(n_fft),
                        )
                        for row in observation
                    ]
                )
                score = _group_balanced_os(reference, observation, output, groups)
                row = {
                    "config_id": (
                        f"noisereduce_{'stationary' if stationary else 'nonstationary'}_"
                        f"p{float(prop_decrease):.2f}_n{int(n_fft)}"
                    ),
                    "stationary": bool(stationary),
                    "prop_decrease": float(prop_decrease),
                    "n_fft": int(n_fft),
                    "validation_os_nrmse": score,
                    "elapsed_seconds": time.perf_counter() - began,
                }
                noisereduce_rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
    selected_wavelet = min(
        wavelet_rows,
        key=lambda row: (
            float(row["validation_os_nrmse"]),
            str(row["wavelet"]),
            str(row["threshold_mode"]),
            float(row["threshold_multiplier"]),
        ),
    )
    selected_noisereduce = min(
        noisereduce_rows,
        key=lambda row: (
            float(row["validation_os_nrmse"]),
            int(not bool(row["stationary"])),
            float(row["prop_decrease"]),
            int(row["n_fft"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "DEVELOPMENT_SELECTION_COMPLETE_TEST_OUTPUTS_NOT_GENERATED",
        "development_path_sha256": _sha256_file(DEVELOPMENT_PATH),
        "tuning_runner_sha256": _sha256_file(RUNNER_PATH),
        "tuning_config_sha256": _sha256_file(CONFIG_PATH),
        "tuning_dependency_sha256": _tuning_dependency_hashes(),
        "package_identity": _package_identity(),
        "selected_validation_windows": 61,
        "validation_conditions": len(reference),
        "source_groups": 61,
        "labels_used": False,
        "selection_metric": config["development"]["selection_metric"],
        "selected_wavelet": selected_wavelet,
        "selected_noisereduce": selected_noisereduce,
        "wavelet_grid": wavelet_rows,
        "noisereduce_grid": noisereduce_rows,
    }
    payload["selection_sha256"] = _canonical_sha256(payload)
    _write_json(SELECTION_PATH, payload, exclusive=True)
    return payload


def freeze_before_evaluation() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    selection = _load_json(SELECTION_PATH)
    _validate_config(config)
    _validate_selection(selection, config)
    if ARTIFACT_DIR.exists() or (OUTPUT_PATH / ".artifacts.staging").exists():
        raise BenchmarkError("evaluation outputs already exist")
    records: dict[str, dict[str, Any]] = {}
    paths = {
        "config": CONFIG_PATH,
        "runner": RUNNER_PATH,
        "tests": PROJECT_ROOT / "tests" / "test_external_denoiser_benchmark_v1.py",
        "protocol_note": Path(__file__).with_name("EXTERNAL_DENOISER_BENCHMARK_V1.md"),
        "dependency_manifest": DEPENDENCY_MANIFEST_PATH,
        "third_party_notice": THIRD_PARTY_NOTICE_PATH,
        "selection": SELECTION_PATH,
        "development": DEVELOPMENT_PATH,
        "development_manifest": PROJECT_ROOT / "data" / "experiment" / "quality_v4_development_v2_manifest.json",
        "development_lock": PROJECT_ROOT / "data" / "experiment" / "quality_v4_development_v2_LOCK.json",
        "thresholds": THRESHOLD_PATH,
        "canonical_waveform_rows": PROJECT_ROOT / "runs" / "hybrid_v2_confirmation_fresh_v1" / "artifacts" / "waveform_rows.csv",
        "runtime_manifest": RUNTIME_MANIFEST,
        "automatic_runtime_source": PROJECT_ROOT / "denoiseapt" / "automatic_runtime.py",
        "inference_source": PROJECT_ROOT / "denoiseapt" / "inference.py",
        "models_source": PROJECT_ROOT / "denoiseapt" / "models.py",
        "experiment_models_source": PROJECT_ROOT / "denoiseapt" / "experiment_models.py",
        "evidence_controller_source": PROJECT_ROOT / "denoiseapt" / "evidence_controller.py",
        "confirmation_evaluation_source": PROJECT_ROOT / "denoiseapt" / "confirmation_evaluation.py",
        "corruptions_source": PROJECT_ROOT / "denoiseapt" / "corruptions.py",
        "rins_architecture": RINS_DIR / "architecture.py",
        "rins_utils": RINS_DIR / "utils.py",
        "rins_demo": RINS_DIR / "demo.ipynb",
        "sensor_trace": TRACE_DIR / "sensor" / "audit_trace.npz",
        "medical_trace": TRACE_DIR / "medical" / "audit_trace.npz",
    }
    for key, path in paths.items():
        records[key] = {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
    import torch

    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXTERNAL_METHOD_OUTPUTS",
        "protocol_id": config["protocol_id"],
        "scientific_scope": config["scientific_scope"],
        "selection_sha256": selection["selection_sha256"],
        "records": records,
        "runtime": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "packages": _package_identity(),
        },
        "rins_t": config["rins_t"],
        "canonical_outputs_absent": True,
    }
    payload["execution_identity_sha256"] = _canonical_sha256(payload)
    _write_json(PREFREEZE_PATH, payload, exclusive=True)
    return payload


def _validate_prefreeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_json(CONFIG_PATH)
    selection = _load_json(SELECTION_PATH)
    receipt = _load_json(PREFREEZE_PATH)
    _validate_config(config)
    _validate_selection(selection, config)
    if receipt.get("status") != "FROZEN_BEFORE_EXTERNAL_METHOD_OUTPUTS":
        raise BenchmarkError("prefreeze status differs")
    for record in receipt.get("records", {}).values():
        path = PROJECT_ROOT / str(record["path"])
        if path.stat().st_size != int(record["bytes"]) or _sha256_file(path) != record["sha256"]:
            raise BenchmarkError(f"prefreeze-bound input differs: {path}")
    if selection.get("selection_sha256") != receipt.get("selection_sha256"):
        raise BenchmarkError("development selection differs from prefreeze")
    identity = dict(receipt)
    expected = identity.pop("execution_identity_sha256")
    if _canonical_sha256(identity) != expected:
        raise BenchmarkError("prefreeze identity differs")
    return config, selection, receipt


def _load_traces() -> dict[str, dict[str, NDArray[Any]]]:
    traces: dict[str, dict[str, NDArray[Any]]] = {}
    required = {
        "metadata_json",
        "reference",
        "observation",
        "labels",
        "centers",
        "scales",
        "condition_id",
        "source_group_id",
        "window_id",
        "series_id",
        "window_kind",
        "family",
        "severity",
        "replicate",
        "identity",
        "target_event_start",
        "target_event_end",
        "output__observation",
        "output__hybrid_v2",
    }
    for domain in DOMAINS:
        with np.load(TRACE_DIR / domain / "audit_trace.npz", allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise BenchmarkError(f"trace schema differs: {domain}")
            traces[domain] = {key: np.asarray(archive[key]) for key in archive.files}
        count = 1200 if domain == "sensor" else 1500
        groups = 20 if domain == "sensor" else 12
        windows = 48 if domain == "sensor" else 60
        trace = traces[domain]
        metadata = json.loads(str(trace["metadata_json"].item()))
        two_dimensional = (
            "reference",
            "observation",
            "labels",
            "output__observation",
            "output__hybrid_v2",
        )
        one_dimensional = (
            "centers",
            "scales",
            "condition_id",
            "source_group_id",
            "window_id",
            "series_id",
            "window_kind",
            "family",
            "severity",
            "replicate",
            "identity",
            "target_event_start",
            "target_event_end",
        )
        if (
            metadata.get("protocol_id") != "denoiseapt-hybrid-v2-confirmation-fresh-v1"
            or metadata.get("domain") != domain
            or int(metadata.get("model_seed", -1)) != 17
            or int(metadata.get("conditions", -1)) != count
            or any(trace[name].shape != (count, 512) for name in two_dimensional)
            or any(trace[name].shape != (count,) for name in one_dimensional)
            or trace["reference"].dtype != np.dtype("float32")
            or trace["observation"].dtype != np.dtype("float32")
            or trace["output__observation"].dtype != np.dtype("float32")
            or trace["output__hybrid_v2"].dtype != np.dtype("float32")
            or trace["labels"].dtype != np.dtype("bool")
            or trace["identity"].dtype != np.dtype("bool")
            or not np.all(np.isfinite(trace["reference"]))
            or not np.all(np.isfinite(trace["observation"]))
            or not np.all(np.isfinite(trace["output__hybrid_v2"]))
            or not np.all(np.isfinite(trace["centers"]))
            or not np.all(np.isfinite(trace["scales"]))
            or not np.all(trace["scales"] > 0.0)
            or not np.array_equal(trace["output__observation"], trace["observation"])
            or len(set(map(str, trace["condition_id"]))) != count
            or len(set(map(str, trace["source_group_id"]))) != groups
            or len(set(map(str, trace["window_id"]))) != windows
            or int(np.sum(trace["identity"].astype(bool))) != windows
        ):
            raise BenchmarkError(f"trace semantics differ: {domain}")
        starts = trace["target_event_start"].astype(int)
        ends = trace["target_event_end"].astype(int)
        if np.any((starts == -1) != (ends == -1)):
            raise BenchmarkError(f"trace event sentinel differs: {domain}")
        for index in np.flatnonzero(starts >= 0):
            start, end = int(starts[index]), int(ends[index])
            if not 0 <= start < end <= 512 or not np.all(trace["labels"][index, start:end]):
                raise BenchmarkError(f"trace target event differs: {domain}/{index}")
    if set(map(str, traces["sensor"]["condition_id"])) & set(
        map(str, traces["medical"]["condition_id"])
    ):
        raise BenchmarkError("condition IDs overlap across domains")
    return traces


def _validate_thresholds(runtime: Any, thresholds: Mapping[str, Any]) -> None:
    from denoiseapt.experiment_models import state_dict_sha256
    from denoiseapt.inference import WindowNormalization

    model_hashes = {
        "A_causal_mlp": state_dict_sha256(runtime.scorer_a),
        "B_causal_conv": state_dict_sha256(runtime.scorer_b),
    }
    specs = {
        spec.witness_id: spec
        for spec in runtime._witnesses(WindowNormalization(center=0.0, scale=1.0))
    }
    if set(specs) != set(CONFIGURED_SCORERS):
        raise BenchmarkError("runtime configured scorer set differs")
    for domain in DOMAINS:
        records = thresholds.get(domain, {})
        if not set(CONFIGURED_SCORERS).issubset(records):
            raise BenchmarkError(f"threshold scorer set differs: {domain}")
        for scorer in CONFIGURED_SCORERS:
            record = records[scorer]
            source_sha = str(record.get("threshold_source_sha256", ""))
            if (
                record.get("witness") != scorer
                or record.get("domain") != domain
                or record.get("role") != "configured"
                or int(record.get("model_seed", -1)) != 17
                or record.get("confirmation_values_used") is not False
                or record.get("model_sha256") != model_hashes[scorer]
                or int(record.get("warmup", -1)) != int(specs[scorer].context_left)
                or not math.isfinite(float(record.get("threshold", math.nan)))
                or len(source_sha) != 64
                or any(character not in "0123456789abcdef" for character in source_sha)
            ):
                raise BenchmarkError(f"threshold semantics differ: {domain}/{scorer}")


def _method_outputs(
    observations: NDArray[np.float32], selection: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, NDArray[np.float32]], dict[str, float]]:
    outputs: dict[str, NDArray[np.float32]] = {"corrupted_input": observations.copy()}
    latencies: dict[str, float] = {"corrupted_input": 0.0}
    began = time.perf_counter()
    outputs["median_filter_w3"] = np.stack([median_filter_w3(row) for row in observations])
    latencies["median_filter_w3"] = (time.perf_counter() - began) * 1000.0 / len(observations)
    selected_wavelet = selection["selected_wavelet"]
    began = time.perf_counter()
    outputs["wavelet_shrinkage"] = np.stack(
        [
            wavelet_shrinkage(
                row,
                wavelet=str(selected_wavelet["wavelet"]),
                threshold_mode=str(selected_wavelet["threshold_mode"]),
                threshold_multiplier=float(selected_wavelet["threshold_multiplier"]),
            )
            for row in observations
        ]
    )
    latencies["wavelet_shrinkage"] = (time.perf_counter() - began) * 1000.0 / len(observations)
    selected_nr = selection["selected_noisereduce"]
    began = time.perf_counter()
    outputs["noisereduce"] = np.stack(
        [
            noisereduce_filter(
                row,
                stationary=bool(selected_nr["stationary"]),
                prop_decrease=float(selected_nr["prop_decrease"]),
                n_fft=int(selected_nr["n_fft"]),
            )
            for row in observations
        ]
    )
    latencies["noisereduce"] = (time.perf_counter() - began) * 1000.0 / len(observations)
    began = time.perf_counter()
    outputs["rins_t"] = run_rins_t(observations, workers=int(config["rins_t"]["workers"]))
    latencies["rins_t"] = (time.perf_counter() - began) * 1000.0 / len(observations)
    return outputs, latencies


def _waveform_metrics(
    reference: NDArray[np.float32],
    observation: NDArray[np.float32],
    output: NDArray[np.float32],
    labels: NDArray[np.bool_],
    scale: float,
) -> dict[str, Any]:
    left = reference.astype(np.float64)
    candidate = output.astype(np.float64)
    error = candidate - left
    rmse = math.sqrt(float(np.mean(np.square(error), dtype=np.float64)))
    mask = np.asarray(labels, dtype=bool)
    anomaly: float | str = ""
    if np.any(mask):
        anomaly = math.sqrt(float(np.mean(np.square(error[mask]), dtype=np.float64))) / scale
    return {
        "os_nrmse": rmse / scale,
        "labelled_anomaly_os_nrmse": anomaly,
        "native_rmse": rmse,
        "mean_abs_correction": float(np.mean(np.abs(candidate - observation.astype(np.float64)))),
        "finite": bool(np.all(np.isfinite(candidate))),
    }


def _metadata(trace: Mapping[str, NDArray[Any]], index: int) -> dict[str, Any]:
    return {
        "domain": str(trace["metadata_json"].item() and json.loads(str(trace["metadata_json"].item()))["domain"]),
        "condition_id": str(trace["condition_id"][index]),
        "source_group_id": str(trace["source_group_id"][index]),
        "window_id": str(trace["window_id"][index]),
        "series_id": str(trace["series_id"][index]),
        "window_kind": str(trace["window_kind"][index]),
        "family": str(trace["family"][index]),
        "severity": float(trace["severity"][index]),
        "replicate": int(trace["replicate"][index]),
        "identity": bool(trace["identity"][index]),
    }


def _canonical_denoiseapt_latencies() -> dict[str, float]:
    path = (
        PROJECT_ROOT
        / "runs"
        / "hybrid_v2_confirmation_fresh_v1"
        / "artifacts"
        / "waveform_rows.csv"
    )
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "hybrid_v2":
                continue
            condition = str(row["condition_id"])
            if condition in values:
                raise BenchmarkError("duplicate canonical DenoiseAPT latency")
            values[condition] = float(row["pipeline_latency_ms"])
    if len(values) != 2700 or any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise BenchmarkError("canonical DenoiseAPT latency grid differs")
    return values


def _aggregate_summary(
    waveform_rows: Sequence[Mapping[str, Any]], witness_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [
            row for row in waveform_rows if row["method"] == method and not bool(row["identity"])
        ]
        endpoint_values: dict[str, float] = {}
        for endpoint, field, event_only in (
            ("overall_os_nrmse", "os_nrmse", False),
            ("anomaly_region_os_nrmse", "labelled_anomaly_os_nrmse", True),
        ):
            domain_means: list[float] = []
            by_domain: dict[str, float] = {}
            for domain in DOMAINS:
                domain_rows = [
                    row
                    for row in method_rows
                    if row["domain"] == domain
                    and (not event_only or row["window_kind"] == "event")
                    and row[field] not in ("", None)
                ]
                group_values: list[float] = []
                for group in sorted({str(row["source_group_id"]) for row in domain_rows}):
                    values = [float(row[field]) for row in domain_rows if row["source_group_id"] == group]
                    group_values.append(float(np.mean(values)))
                if not group_values:
                    raise BenchmarkError(f"empty aggregate: {method}/{endpoint}/{domain}")
                by_domain[domain] = float(np.mean(group_values))
                domain_means.append(by_domain[domain])
            endpoint_values[endpoint] = float(np.mean(domain_means))
            endpoint_values[f"{endpoint}_by_domain"] = by_domain  # type: ignore[assignment]
        counts = [row for row in witness_rows if row["method"] == method]
        denominator = sum(int(row["denoising_erasure_denominator"]) for row in counts)
        erased = sum(int(row["denoising_erasure_numerator"]) for row in counts)
        new = sum(int(row["output_only_evidence_intervals"]) for row in counts)
        methods[method] = {
            **endpoint_values,
            "evidence_retained_numerator": denominator - erased,
            "evidence_retention_denominator": denominator,
            "evidence_retained_percent": (
                None if method == "corrupted_input" else 100.0 * (denominator - erased) / denominator
            ),
            "evidence_lost": erased,
            "new_output_only_intervals": new,
        }
    return methods


def _assert_existing_row_parity(methods: Mapping[str, Mapping[str, Any]]) -> None:
    """Prove that the new evaluator reproduces three already-audited rows."""

    expected = {
        "corrupted_input": {
            "overall_os_nrmse": 0.2088335577294494,
            "anomaly_region_os_nrmse": 0.20114505652829967,
            "evidence_lost": 0,
            "evidence_retention_denominator": 2166,
            "new_output_only_intervals": 0,
        },
        "median_filter_w3": {
            "overall_os_nrmse": 0.15482905507087708,
            "anomaly_region_os_nrmse": 0.18997987364340993,
            "evidence_lost": 505,
            "evidence_retention_denominator": 2166,
            "new_output_only_intervals": 0,
        },
        "denoiseapt": {
            "overall_os_nrmse": 0.18634294227594744,
            "anomaly_region_os_nrmse": 0.18559590558093653,
            "evidence_lost": 30,
            "evidence_retention_denominator": 2166,
            "new_output_only_intervals": 0,
        },
    }
    for method, fields in expected.items():
        observed = methods[method]
        for field, value in fields.items():
            if isinstance(value, float):
                if not math.isclose(float(observed[field]), value, rel_tol=0.0, abs_tol=1e-12):
                    raise BenchmarkError(f"existing metric parity failed: {method}/{field}")
            elif int(observed[field]) != value:
                raise BenchmarkError(f"existing count parity failed: {method}/{field}")


def evaluate() -> dict[str, Any]:
    config, selection, receipt = _validate_prefreeze()
    if SUMMARY_PATH.exists():
        return _load_json(SUMMARY_PATH)
    staging = OUTPUT_PATH / ".artifacts.staging"
    if ARTIFACT_DIR.exists() or staging.exists():
        raise BenchmarkError("partial benchmark artifact directory exists")
    from denoiseapt.automatic_runtime import AutomaticPreservationRuntime
    from denoiseapt.confirmation_evaluation import detector_condition_counts
    from denoiseapt.inference import WindowNormalization

    runtime = AutomaticPreservationRuntime(PROJECT_ROOT)
    thresholds = _load_json(THRESHOLD_PATH)["by_domain"]
    _validate_thresholds(runtime, thresholds)
    traces = _load_traces()
    denoiseapt_latency = _canonical_denoiseapt_latencies()
    trace_condition_ids = {
        str(value) for domain in DOMAINS for value in traces[domain]["condition_id"]
    }
    if set(denoiseapt_latency) != trace_condition_ids:
        raise BenchmarkError("canonical latency condition grid differs from traces")
    all_outputs: dict[str, NDArray[np.float32]] = {}
    waveform_rows: list[dict[str, Any]] = []
    witness_rows: list[dict[str, Any]] = []
    latency_by_domain: dict[str, dict[str, float]] = {}
    for domain in DOMAINS:
        trace = traces[domain]
        observations = np.asarray(trace["observation"], dtype=np.float32)
        references = np.asarray(trace["reference"], dtype=np.float32)
        labels = np.asarray(trace["labels"], dtype=bool)
        centers = np.asarray(trace["centers"], dtype=np.float64)
        scales = np.asarray(trace["scales"], dtype=np.float64)
        outputs, latencies = _method_outputs(observations, selection, config)
        outputs["denoiseapt"] = np.asarray(trace["output__hybrid_v2"], dtype=np.float32)
        domain_condition_ids = [str(value) for value in trace["condition_id"]]
        latencies["denoiseapt"] = float(
            np.mean([denoiseapt_latency[value] for value in domain_condition_ids])
        )
        if set(outputs) != set(METHODS):
            raise BenchmarkError("method output set differs")
        latency_by_domain[domain] = latencies
        for method, output in outputs.items():
            if output.shape != observations.shape or not np.all(np.isfinite(output)):
                raise BenchmarkError(f"invalid output grid: {domain}/{method}")
            all_outputs[f"{domain}__{method}"] = output
        for index in range(len(observations)):
            normalization = WindowNormalization(
                center=float(centers[index]), scale=float(scales[index])
            )
            specs = {spec.witness_id: spec for spec in runtime._witnesses(normalization)}
            reference_scores = {
                scorer: specs[scorer].scorer(references[index]) for scorer in CONFIGURED_SCORERS
            }
            observation_scores = {
                scorer: specs[scorer].scorer(observations[index]) for scorer in CONFIGURED_SCORERS
            }
            for method in METHODS:
                output = outputs[method]
                output_scores = {
                    scorer: specs[scorer].scorer(output[index]) for scorer in CONFIGURED_SCORERS
                }
                metadata = _metadata(trace, index)
                waveform_rows.append(
                    {
                        **metadata,
                        "method": method,
                        **_waveform_metrics(
                            references[index], observations[index], output[index], labels[index], float(scales[index])
                        ),
                        "pipeline_latency_ms": (
                            denoiseapt_latency[str(trace["condition_id"][index])]
                            if method == "denoiseapt"
                            else float(latencies[method])
                        ),
                        "output_sha256": _array_sha256(output[index]),
                    }
                )
                start = int(trace["target_event_start"][index])
                end = int(trace["target_event_end"][index])
                target = None if start < 0 else (start, end)
                for scorer in CONFIGURED_SCORERS:
                    record = thresholds[domain][scorer]
                    counts = detector_condition_counts(
                        labels=labels[index],
                        target_event_offset=target,
                        reference_scores=reference_scores[scorer],
                        observation_scores=observation_scores[scorer],
                        output_scores=output_scores[scorer],
                        threshold=float(record["threshold"]),
                        warmup=int(specs[scorer].context_left),
                        matching_tolerance=int(runtime.controller_config.fabrication_dilation),
                    )
                    witness_rows.append(
                        {
                            **metadata,
                            "method": method,
                            "scorer": scorer,
                            "threshold": float(record["threshold"]),
                            "threshold_source_sha256": str(record["threshold_source_sha256"]),
                            **counts,
                        }
                    )
        print(f"completed domain={domain} methods={len(METHODS)} conditions={len(observations)}", flush=True)
    if len(waveform_rows) != 2700 * len(METHODS) or len(witness_rows) != 2700 * len(METHODS) * 2:
        raise BenchmarkError("benchmark row cardinality differs")
    method_summary = _aggregate_summary(waveform_rows, witness_rows)
    _assert_existing_row_parity(method_summary)
    output_arrays: dict[str, Any] = {
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol_id": config["protocol_id"],
                    "methods": list(METHODS),
                    "execution_identity_sha256": receipt["execution_identity_sha256"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    }
    output_arrays.update(all_outputs)
    staging.mkdir(parents=True, exist_ok=False)
    staged_outputs = staging / OUTPUTS_PATH.name
    staged_waveform = staging / WAVEFORM_ROWS_PATH.name
    staged_witness = staging / WITNESS_ROWS_PATH.name
    staged_summary = staging / SUMMARY_PATH.name
    temporary_npz = staging / f".{OUTPUTS_PATH.name}.tmp.npz"
    np.savez_compressed(temporary_npz, **output_arrays)
    temporary_npz.replace(staged_outputs)
    _write_rows(staged_waveform, waveform_rows)
    _write_rows(staged_witness, witness_rows)
    payload = {
        "schema_version": 1,
        "status": "COMPLETE_RETROSPECTIVE_DESCRIPTIVE_BENCHMARK",
        "protocol_id": config["protocol_id"],
        "scientific_scope": config["scientific_scope"],
        "execution_identity_sha256": receipt["execution_identity_sha256"],
        "development_selection": {
            "selection_sha256": selection["selection_sha256"],
            "selected_wavelet": selection["selected_wavelet"],
            "selected_noisereduce": selection["selected_noisereduce"],
        },
        "methods": method_summary,
        "latency_ms_per_window_by_domain": latency_by_domain,
        "cardinality": {
            "conditions": 2700,
            "corrupted_conditions": 2592,
            "source_groups": 32,
            "waveform_rows": len(waveform_rows),
            "witness_rows": len(witness_rows),
        },
        "artifacts": {
            path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for path in (staged_outputs, staged_waveform, staged_witness)
        },
        "limitations": [
            "The comparator roster was selected after the panel had been opened; all comparisons are retrospective and descriptive.",
            "RINS-T used an official-architecture/demo-recipe adaptation from a repository with no software license at the pinned commit; its code is not redistributed by this benchmark.",
            "Configured-scorer evidence retention is not physical anomaly truth or detector-independent preservation.",
            "ECG-specific methods were excluded because the mixed Medical track includes non-ECG data and lacks sampling-rate and lead metadata.",
        ],
    }
    payload["summary_sha256"] = _canonical_sha256(payload)
    _write_json(staged_summary, payload, exclusive=True)
    staging.replace(ARTIFACT_DIR)
    return payload


def _read_waveform_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["identity"] = str(raw["identity"]).lower() == "true"
            row["finite"] = str(raw["finite"]).lower() == "true"
            row["severity"] = float(raw["severity"])
            row["replicate"] = int(raw["replicate"])
            for field in (
                "os_nrmse",
                "native_rmse",
                "mean_abs_correction",
                "pipeline_latency_ms",
            ):
                row[field] = float(raw[field])
            row["labelled_anomaly_os_nrmse"] = (
                ""
                if raw["labelled_anomaly_os_nrmse"] == ""
                else float(raw["labelled_anomaly_os_nrmse"])
            )
            rows.append(row)
    return rows


def _read_witness_rows(path: Path) -> list[dict[str, Any]]:
    count_fields = (
        "target_labelled_event_opportunities",
        "target_reference_detectable_event_opportunities",
        "target_observation_detectable_event_opportunities",
        "target_output_detectable_event_opportunities",
        "denoising_erasure_numerator",
        "denoising_erasure_denominator",
        "observation_evidence_intervals",
        "output_evidence_intervals",
        "output_only_evidence_intervals",
    )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            row["identity"] = str(raw["identity"]).lower() == "true"
            row["severity"] = float(raw["severity"])
            row["replicate"] = int(raw["replicate"])
            row["threshold"] = float(raw["threshold"])
            for field in count_fields:
                row[field] = int(raw[field])
            rows.append(row)
    return rows


def audit() -> dict[str, Any]:
    _, _, receipt = _validate_prefreeze()
    summary = _load_json(SUMMARY_PATH)
    value = dict(summary)
    expected_summary_sha = value.pop("summary_sha256")
    if _canonical_sha256(value) != expected_summary_sha:
        raise BenchmarkError("summary self-hash differs")
    if summary.get("execution_identity_sha256") != receipt.get("execution_identity_sha256"):
        raise BenchmarkError("summary execution identity differs")
    for name, record in summary["artifacts"].items():
        path = ARTIFACT_DIR / name
        if path.stat().st_size != int(record["bytes"]) or _sha256_file(path) != record["sha256"]:
            raise BenchmarkError(f"artifact differs: {name}")
    waveform_rows = _read_waveform_rows(WAVEFORM_ROWS_PATH)
    witness_rows = _read_witness_rows(WITNESS_ROWS_PATH)
    if len(waveform_rows) != 2700 * len(METHODS) or len(witness_rows) != 2700 * len(METHODS) * 2:
        raise BenchmarkError("audited CSV cardinality differs")
    waveform_keys = {
        (str(row["domain"]), str(row["condition_id"]), str(row["method"]))
        for row in waveform_rows
    }
    witness_keys = {
        (
            str(row["domain"]),
            str(row["condition_id"]),
            str(row["method"]),
            str(row["scorer"]),
        )
        for row in witness_rows
    }
    if len(waveform_keys) != len(waveform_rows) or len(witness_keys) != len(witness_rows):
        raise BenchmarkError("audited CSV keys are duplicated")
    recomputed_methods = _aggregate_summary(waveform_rows, witness_rows)
    _assert_existing_row_parity(recomputed_methods)
    if _canonical_sha256(recomputed_methods) != _canonical_sha256(summary["methods"]):
        raise BenchmarkError("summary methods differ from CSV recomputation")
    traces = _load_traces()
    waveform_by_key = {
        (str(row["domain"]), str(row["condition_id"]), str(row["method"])): row
        for row in waveform_rows
    }
    with np.load(OUTPUTS_PATH, allow_pickle=False) as archive:
        expected = {"metadata_json"} | {f"{domain}__{method}" for domain in DOMAINS for method in METHODS}
        if set(archive.files) != expected:
            raise BenchmarkError("output NPZ schema differs")
        for domain in DOMAINS:
            count = 1200 if domain == "sensor" else 1500
            for method in METHODS:
                values = np.asarray(archive[f"{domain}__{method}"], dtype=np.float32)
                if values.shape != (count, 512) or not np.all(np.isfinite(values)):
                    raise BenchmarkError(f"audited output differs: {domain}/{method}")
                if method == "corrupted_input" and not np.array_equal(
                    values, traces[domain]["observation"]
                ):
                    raise BenchmarkError(f"corrupted-input source equality differs: {domain}")
                if method == "denoiseapt" and not np.array_equal(
                    values, traces[domain]["output__hybrid_v2"]
                ):
                    raise BenchmarkError(f"DenoiseAPT source equality differs: {domain}")
                for index, condition in enumerate(traces[domain]["condition_id"]):
                    row = waveform_by_key[(domain, str(condition), method)]
                    if row["output_sha256"] != _array_sha256(values[index]):
                        raise BenchmarkError(f"waveform output hash differs: {domain}/{method}/{index}")
    return {
        "status": "AUDIT_PASS",
        "execution_identity_sha256": receipt["execution_identity_sha256"],
        "summary_sha256": summary["summary_sha256"],
        "artifacts": summary["artifacts"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum((args.tune, args.freeze, args.evaluate, args.audit)) != 1:
        raise SystemExit("choose exactly one of --tune, --freeze, --evaluate, --audit")
    if args.tune:
        result = tune_development()
    elif args.freeze:
        result = freeze_before_evaluation()
    elif args.evaluate:
        result = evaluate()
    else:
        result = audit()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
