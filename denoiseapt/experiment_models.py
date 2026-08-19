"""Models and utilities used only by the frozen publication protocol.

This module is deliberately separate from the live demonstration checkpoint.
It adds two detectors that are never used by the generator preservation loss:
an independently initialized causal convolutional forecaster and a
dependency-light spectral-residual detector.  Their purpose is to test whether
preservation transfers beyond the causal MLP scorer used for training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CausalConvConfig:
    """Serializable configuration for :class:`CausalConvScorer`."""

    hidden_channels: int = 24
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least two")
        if not self.dilations or any(dilation < 1 for dilation in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def warmup(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CausalConvConfig":
        data = dict(values)
        if "dilations" in data:
            data["dilations"] = tuple(int(value) for value in data["dilations"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dilations"] = list(self.dilations)
        return data


class _LeftCausalConv(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel: int, dilation: int):
        super().__init__()
        self.left_padding = dilation * (kernel - 1)
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernel,
            dilation=dilation,
            padding=0,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(F.pad(inputs, (self.left_padding, 0), mode="replicate"))


class CausalConvScorer(nn.Module):
    """One-step causal convolutional forecaster used as transfer detector B.

    The input is shifted right before entering the network.  Consequently, the
    prediction at timestamp ``t`` cannot depend on the value at ``t`` or any
    future value.  Initial timestamps inside the receptive-field warm-up are
    assigned score zero and excluded from threshold calibration.
    """

    def __init__(self, config: CausalConvConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalConvConfig()
        layers: list[nn.Module] = []
        channels = 1
        for dilation in self.config.dilations:
            layers.extend(
                [
                    _LeftCausalConv(
                        channels,
                        self.config.hidden_channels,
                        self.config.kernel_size,
                        dilation,
                    ),
                    nn.SiLU(),
                ]
            )
            if self.config.dropout:
                layers.append(nn.Dropout(self.config.dropout))
            channels = self.config.hidden_channels
        self.network = nn.Sequential(*layers)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, signal: Tensor) -> Tensor:
        _validate_signal(signal)
        delayed = torch.cat((signal[..., :1], signal[..., :-1]), dim=-1)
        return self.output(self.network(delayed))

    def anomaly_score(self, signal: Tensor) -> Tensor:
        predictions = self(signal)
        scores = (signal - predictions).square()
        scores = scores.clone()
        scores[..., : self.config.warmup] = 0.0
        return scores

    def freeze(self) -> "CausalConvScorer":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze(self) -> "CausalConvScorer":
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self


@dataclass(frozen=True)
class SpectralResidualConfig:
    """Configuration for transfer detector C."""

    log_spectrum_window: int = 5
    saliency_window: int = 21
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.log_spectrum_window < 1 or self.saliency_window < 3:
            raise ValueError("spectral-residual windows are too short")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpectralResidualDetector:
    """Deterministic NumPy spectral-residual anomaly scorer.

    This is a compact implementation for cross-detector auditing, not a claim
    of reproducing a particular benchmark package.  It has no learned
    parameters and is not used for model selection.
    """

    def __init__(self, config: SpectralResidualConfig | None = None) -> None:
        self.config = config or SpectralResidualConfig()

    def score(self, signal: ArrayLike) -> NDArray[np.float32]:
        values = np.asarray(signal, dtype=np.float64)
        if values.ndim != 1 or values.size < 32 or not np.all(np.isfinite(values)):
            raise ValueError("signal must contain at least 32 finite samples")
        spectrum = np.fft.fft(values)
        amplitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        log_amplitude = np.log(amplitude + self.config.epsilon)
        average = _moving_average(log_amplitude, self.config.log_spectrum_window)
        residual = log_amplitude - average
        saliency = np.abs(np.fft.ifft(np.exp(residual + 1j * phase)))
        local_mean = _moving_average(saliency, self.config.saliency_window)
        local_second = _moving_average(saliency * saliency, self.config.saliency_window)
        local_std = np.sqrt(np.maximum(local_second - local_mean * local_mean, 0.0))
        scores = np.maximum(saliency - local_mean, 0.0) / (
            local_std + self.config.epsilon
        )
        return np.asarray(scores, dtype=np.float32)


def torch_detector_scores(
    model: nn.Module,
    windows: ArrayLike,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> NDArray[np.float32]:
    """Score a fixed window matrix without enabling stochastic layers."""

    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 32 or not np.all(np.isfinite(values)):
        raise ValueError("windows must have shape [count, time] with finite values")
    target = torch.device(device)
    model = model.to(target).eval()
    outputs: list[NDArray[np.float32]] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size, None, :]).to(target)
            score = model.anomaly_score(batch)  # type: ignore[attr-defined]
            outputs.append(score[:, 0].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def train_causal_detector(
    model: nn.Module,
    normal_windows: ArrayLike,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Train scorer A or B only on normal, independently normalized windows."""

    values = np.asarray(normal_windows, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] < 32:
        raise ValueError("normal_windows must have shape [count, time]")
    if not np.all(np.isfinite(values)):
        raise ValueError("normal_windows contain non-finite values")
    normalized = np.stack([robust_normalize(window) for window in values])
    target = torch.device(device)
    model = model.to(target)
    if hasattr(model, "unfreeze"):
        model.unfreeze()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    torch_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(normalized[:, None, :]))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch_generator,
    )
    warmup = _detector_warmup(model)
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for (signals,) in loader:
            signals = signals.to(target)
            predictions = model(signals)
            loss = F.mse_loss(predictions[..., warmup:], signals[..., warmup:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += float(loss.detach()) * len(signals)
            count += len(signals)
        history.append({"epoch": float(epoch + 1), "forecast_mse": total / max(count, 1)})
    if hasattr(model, "freeze"):
        model.freeze()
    else:
        model.eval()
    return history


def robust_normalize(signal: ArrayLike) -> NDArray[np.float32]:
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be one-dimensional and finite")
    center = float(np.median(values))
    q25, q75 = np.percentile(values, [25.0, 75.0])
    scale = max(float((q75 - q25) / 1.349), float(np.std(values)), 1e-4)
    return ((values - center) / scale).astype(np.float32)


def normalize_pair(
    reference: ArrayLike, observation: ArrayLike
) -> tuple[NDArray[np.float32], NDArray[np.float32], float, float]:
    """Normalize a controlled pair using statistics from the observation."""

    clean = np.asarray(reference, dtype=np.float32)
    noisy = np.asarray(observation, dtype=np.float32)
    if clean.shape != noisy.shape or clean.ndim != 1:
        raise ValueError("reference and observation must be matching vectors")
    center = float(np.median(noisy))
    q25, q75 = np.percentile(noisy, [25.0, 75.0])
    scale = max(float((q75 - q25) / 1.349), float(np.std(noisy)), 1e-4)
    return (
        ((clean - center) / scale).astype(np.float32),
        ((noisy - center) / scale).astype(np.float32),
        center,
        scale,
    )


def state_dict_sha256(model: nn.Module) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in deterministic order."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _detector_warmup(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        if hasattr(config, "context_length"):
            return int(config.context_length)
        if hasattr(config, "warmup"):
            return int(config.warmup)
    return 1


def _moving_average(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    width = max(1, min(int(width), len(values)))
    if width % 2 == 0 and width > 1:
        width -= 1
    if width == 1:
        return values.copy()
    pad = width // 2
    padded = np.pad(values, pad, mode="reflect")
    return np.convolve(padded, np.full(width, 1.0 / width), mode="valid")


def _validate_signal(signal: Tensor) -> None:
    if signal.ndim != 3 or signal.shape[1] != 1 or signal.shape[-1] < 32:
        raise ValueError("expected [batch, 1, time] with time >= 32")
    if not signal.is_floating_point():
        raise TypeError("signal tensor must be floating point")
