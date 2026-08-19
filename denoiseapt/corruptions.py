"""Deterministic synthetic measurement-corruption operators for the demo.

These operators are used only in controlled mode, where the original signal is
retained for evaluation.  They are not intended to claim that every real-world
noise process belongs to one of these families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


CorruptionKind = Literal["none", "gaussian", "impulse", "drift", "mixed"]


@dataclass(frozen=True)
class CorruptionResult:
    clean: NDArray[np.float32]
    corrupted: NDArray[np.float32]
    residual: NDArray[np.float32]
    kind: CorruptionKind
    severity: float
    seed: int
    parameters: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("clean", "corrupted", "residual"):
            data.pop(key)
        return data


def robust_signal_scale(signal: ArrayLike) -> float:
    """Estimate signal scale without allowing a flat signal to produce NaNs."""

    values = _as_signal(signal)
    q25, q75 = np.percentile(values, [25.0, 75.0])
    robust_std = float((q75 - q25) / 1.349)
    ordinary_std = float(np.std(values))
    return max(robust_std, ordinary_std, 1e-4)


def gaussian_corruption(
    signal: ArrayLike, severity: float, rng: np.random.Generator
) -> tuple[NDArray[np.float32], dict[str, float]]:
    values = _as_signal(signal)
    severity = _validate_severity(severity)
    sigma = 0.25 * severity * robust_signal_scale(values)
    noise = rng.normal(0.0, sigma, size=values.shape)
    return (values + noise).astype(np.float32), {"sigma": float(sigma)}


def impulse_corruption(
    signal: ArrayLike, severity: float, rng: np.random.Generator
) -> tuple[NDArray[np.float32], dict[str, float | int]]:
    values = _as_signal(signal)
    severity = _validate_severity(severity)
    probability = 0.08 * severity
    impulse_mask = rng.random(values.shape[0]) < probability
    amplitude = (1.0 + 2.0 * severity) * robust_signal_scale(values)
    impulses = rng.choice(np.asarray([-1.0, 1.0]), size=values.shape[0])
    impulses *= amplitude * rng.uniform(0.65, 1.0, size=values.shape[0])
    corrupted = values + impulse_mask * impulses
    return corrupted.astype(np.float32), {
        "probability": float(probability),
        "amplitude": float(amplitude),
        "count": int(impulse_mask.sum()),
    }


def drift_corruption(
    signal: ArrayLike, severity: float, rng: np.random.Generator
) -> tuple[NDArray[np.float32], dict[str, float]]:
    values = _as_signal(signal)
    severity = _validate_severity(severity)
    scale = robust_signal_scale(values)
    length = values.shape[0]
    normalized_time = np.linspace(0.0, 1.0, length, dtype=np.float64)
    cycles = float(rng.uniform(0.35, 1.25))
    phase = float(rng.uniform(-np.pi, np.pi))
    sinusoid = np.sin(2.0 * np.pi * cycles * normalized_time + phase)
    walk = np.cumsum(rng.normal(0.0, 1.0, length))
    walk -= np.linspace(walk[0], walk[-1], length)
    walk /= max(float(np.std(walk)), 1e-6)
    amplitude = 0.50 * severity * scale
    drift = amplitude * (0.8 * sinusoid + 0.2 * walk)
    return (values + drift).astype(np.float32), {
        "amplitude": float(amplitude),
        "cycles": cycles,
        "phase": phase,
    }


def apply_measurement_corruption(
    signal: ArrayLike,
    kind: CorruptionKind = "gaussian",
    severity: float = 0.35,
    seed: int = 7,
) -> CorruptionResult:
    """Apply one reproducible corruption family to a one-dimensional signal."""

    clean = _as_signal(signal)
    severity = _validate_severity(severity)
    if kind not in {"none", "gaussian", "impulse", "drift", "mixed"}:
        raise ValueError(f"Unsupported corruption kind: {kind!r}")
    rng = np.random.default_rng(int(seed))

    parameters: dict[str, Any]
    if kind == "none" or severity == 0:
        corrupted = clean.copy()
        parameters = {}
    elif kind == "gaussian":
        corrupted, parameters = gaussian_corruption(clean, severity, rng)
    elif kind == "impulse":
        corrupted, parameters = impulse_corruption(clean, severity, rng)
    elif kind == "drift":
        corrupted, parameters = drift_corruption(clean, severity, rng)
    else:
        # Independent generators avoid changing all mixed-corruption traces if
        # one component later consumes a different number of random values.
        component_seeds = rng.integers(0, np.iinfo(np.int32).max, size=3)
        gaussian, gaussian_parameters = gaussian_corruption(
            clean, severity * 0.65, np.random.default_rng(int(component_seeds[0]))
        )
        impulse, impulse_parameters = impulse_corruption(
            gaussian, severity * 0.65, np.random.default_rng(int(component_seeds[1]))
        )
        corrupted, drift_parameters = drift_corruption(
            impulse, severity * 0.55, np.random.default_rng(int(component_seeds[2]))
        )
        parameters = {
            "gaussian": gaussian_parameters,
            "impulse": impulse_parameters,
            "drift": drift_parameters,
        }

    corrupted = np.asarray(corrupted, dtype=np.float32)
    return CorruptionResult(
        clean=clean.copy(),
        corrupted=corrupted,
        residual=(corrupted - clean).astype(np.float32),
        kind=kind,
        severity=severity,
        seed=int(seed),
        parameters=parameters,
    )


def _as_signal(signal: ArrayLike) -> NDArray[np.float32]:
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Corruption operators require a one-dimensional signal.")
    if values.size < 8:
        raise ValueError("A signal must contain at least eight timestamps.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signal values must all be finite.")
    return values


def _validate_severity(severity: float) -> float:
    value = float(severity)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Corruption severity must be a finite value in [0, 1].")
    return value

