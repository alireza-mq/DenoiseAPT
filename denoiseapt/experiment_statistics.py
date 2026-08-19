"""Series-level statistics and predeclared claim gates for benchmark experiments.

The source time series, rather than a window, corruption draw, or timestamp, is
the independent sampling unit.  Callers must first average repeated conditions
within a source series and then pass one paired value per series to this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate and percentile cluster-bootstrap confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int
    bootstrap_replicates: int
    seed: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimGate:
    """Outcome of the predeclared preservation-with-restoration claim gate."""

    supported: bool
    primary_preservation_pass: bool
    restoration_noninferiority_pass: bool
    transfer_direction_pass: bool
    preservation_difference: IntervalEstimate
    relative_rmse_increase: IntervalEstimate
    transfer_difference: IntervalEstimate | None
    rmse_noninferiority_margin: float
    interpretation: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["preservation_difference"] = self.preservation_difference.to_dict()
        result["relative_rmse_increase"] = self.relative_rmse_increase.to_dict()
        result["transfer_difference"] = (
            None if self.transfer_difference is None else self.transfer_difference.to_dict()
        )
        return result


def paired_cluster_bootstrap(
    left: Mapping[str, float] | Sequence[float],
    right: Mapping[str, float] | Sequence[float],
    *,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 2026,
    relative_to_left: bool = False,
) -> IntervalEstimate:
    """Estimate ``mean(right - left)`` with a paired cluster bootstrap.

    If ``relative_to_left`` is true, each paired contribution is
    ``(right-left)/abs(left)``.  Non-finite pairs and relative pairs whose left
    value is numerically zero are excluded explicitly.  Mapping inputs are
    aligned by identical source-series keys and reject missing clusters.
    """

    x, y = _paired_values(left, right)
    valid = np.isfinite(x) & np.isfinite(y)
    if relative_to_left:
        valid &= np.abs(x) > np.finfo(np.float64).eps
    x, y = x[valid], y[valid]
    if x.size < 2:
        raise ValueError("At least two finite paired source-series values are required.")
    contributions = (y - x) / np.abs(x) if relative_to_left else y - x
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    rng = np.random.default_rng(int(seed))
    sampled = rng.integers(0, contributions.size, size=(replicates, contributions.size))
    estimates = contributions[sampled].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return IntervalEstimate(
        estimate=float(np.mean(contributions)),
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        clusters=int(contributions.size),
        bootstrap_replicates=int(replicates),
        seed=int(seed),
    )


def paired_sign_permutation_test(
    left: Mapping[str, float] | Sequence[float],
    right: Mapping[str, float] | Sequence[float],
    *,
    alternative: str = "two-sided",
    replicates: int = 100_000,
    seed: int = 2026,
) -> float:
    """Paired sign-flip permutation p-value at the source-series level."""

    x, y = _paired_values(left, right)
    differences = y - x
    differences = differences[np.isfinite(differences)]
    if differences.size < 2:
        raise ValueError("At least two finite paired source-series values are required.")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    observed = float(np.mean(differences))
    rng = np.random.default_rng(int(seed))
    exceedances = 0
    completed = 0
    # Batching bounds peak memory for large, reproducible Monte Carlo runs.
    while completed < replicates:
        count = min(4096, replicates - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, differences.size))
        null = (signs * differences).mean(axis=1)
        if alternative == "two-sided":
            exceedances += int(np.sum(np.abs(null) >= abs(observed)))
        elif alternative == "less":
            exceedances += int(np.sum(null <= observed))
        else:
            exceedances += int(np.sum(null >= observed))
        completed += count
    # The plus-one correction prevents a zero Monte Carlo p-value.
    return float((exceedances + 1) / (replicates + 1))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm family-wise-error adjusted p-values in original order."""

    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite values in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    adjusted_sorted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty_like(values)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def preservation_claim_gate(
    cgan_erasure: Mapping[str, float] | Sequence[float],
    denoiseapt_erasure: Mapping[str, float] | Sequence[float],
    cgan_rmse: Mapping[str, float] | Sequence[float],
    denoiseapt_rmse: Mapping[str, float] | Sequence[float],
    *,
    transfer_cgan_erasure: Mapping[str, float] | Sequence[float] | None = None,
    transfer_denoiseapt_erasure: Mapping[str, float] | Sequence[float] | None = None,
    rmse_noninferiority_margin: float = 0.05,
    confidence: float = 0.95,
    replicates: int = 10_000,
    seed: int = 2026,
) -> ClaimGate:
    """Apply the predeclared efficacy gate without inspecting individual cases.

    Lower anomaly-erasure and lower RMSE are preferable.  The preservation
    endpoint passes only when the complete CI for DenoiseAPT minus cGAN is below
    zero.  Restoration passes when the CI upper bound for relative RMSE increase
    is no larger than the predeclared margin.  If a transfer set is supplied,
    its point estimate must have the same (negative) direction; it is not used
    to tune any model or threshold.
    """

    if not math.isfinite(rmse_noninferiority_margin) or rmse_noninferiority_margin < 0:
        raise ValueError("rmse_noninferiority_margin must be finite and non-negative")
    preservation = paired_cluster_bootstrap(
        cgan_erasure,
        denoiseapt_erasure,
        confidence=confidence,
        replicates=replicates,
        seed=seed,
    )
    rmse = paired_cluster_bootstrap(
        cgan_rmse,
        denoiseapt_rmse,
        confidence=confidence,
        replicates=replicates,
        seed=seed + 1,
        relative_to_left=True,
    )
    transfer: IntervalEstimate | None = None
    transfer_pass = True
    if (transfer_cgan_erasure is None) != (transfer_denoiseapt_erasure is None):
        raise ValueError("Both transfer erasure inputs must be supplied together.")
    if transfer_cgan_erasure is not None and transfer_denoiseapt_erasure is not None:
        transfer = paired_cluster_bootstrap(
            transfer_cgan_erasure,
            transfer_denoiseapt_erasure,
            confidence=confidence,
            replicates=replicates,
            seed=seed + 2,
        )
        transfer_pass = transfer.estimate < 0.0
    preservation_pass = preservation.upper < 0.0
    restoration_pass = rmse.upper <= rmse_noninferiority_margin
    supported = preservation_pass and restoration_pass and transfer_pass
    if supported:
        interpretation = (
            "The predeclared held-out evidence supports a qualified preservation claim "
            "subject to the reported detector, corruption, and dataset scope."
        )
    else:
        interpretation = (
            "The predeclared efficacy claim is not supported; report DenoiseAPT as an "
            "interactive auditing/control artifact and disclose the failed gate."
        )
    return ClaimGate(
        supported=supported,
        primary_preservation_pass=preservation_pass,
        restoration_noninferiority_pass=restoration_pass,
        transfer_direction_pass=transfer_pass,
        preservation_difference=preservation,
        relative_rmse_increase=rmse,
        transfer_difference=transfer,
        rmse_noninferiority_margin=float(rmse_noninferiority_margin),
        interpretation=interpretation,
    )


def _paired_values(
    left: Mapping[str, float] | Sequence[float],
    right: Mapping[str, float] | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError("left and right must both be mappings or both be sequences")
        if set(left) != set(right):
            missing_left = sorted(set(right) - set(left))
            missing_right = sorted(set(left) - set(right))
            raise ValueError(
                f"Paired cluster keys differ; missing_left={missing_left}, "
                f"missing_right={missing_right}"
            )
        keys = sorted(left)
        return (
            np.asarray([left[key] for key in keys], dtype=np.float64),
            np.asarray([right[key] for key in keys], dtype=np.float64),
        )
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Paired sequences must be one-dimensional and have equal shape.")
    return x, y
