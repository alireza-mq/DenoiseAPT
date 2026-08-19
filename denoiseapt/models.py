"""Compact PyTorch models used by the DenoiseAPT demonstration.

The models deliberately favour predictable CPU inference over architectural
novelty.  They implement the components used by the demonstration: a temporal
U-Net generator, a conditional temporal PatchGAN discriminator, and a frozen
causal forecaster whose squared prediction error is used as anomaly evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _normalization_groups(channels: int, requested: int) -> int:
    """Return the largest valid GroupNorm group count up to ``requested``."""

    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class GeneratorConfig:
    """Serializable configuration for :class:`TemporalUNetGenerator`."""

    in_channels: int = 1
    base_channels: int = 16
    depth: int = 3
    max_channels: int = 128
    dropout: float = 0.15
    residual_scale: float = 1.0
    normalization_groups: int = 4

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GeneratorConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscriminatorConfig:
    """Serializable configuration for :class:`TemporalPatchDiscriminator`."""

    signal_channels: int = 1
    base_channels: int = 16
    layers: int = 3
    max_channels: int = 128
    normalization_groups: int = 4

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DiscriminatorConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecasterConfig:
    """Serializable configuration for the frozen causal anomaly scorer."""

    context_length: int = 16
    hidden_channels: int = 32
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ForecasterConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_groups = _normalization_groups(out_channels, groups)
        modules: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
        ]
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class _DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.project = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.refine = _ConvBlock(
            out_channels + skip_channels,
            out_channels,
            stride=1,
            groups=groups,
            dropout=dropout,
        )

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        # Explicit sizing supports windows that are not powers of two.
        resized = F.interpolate(inputs, size=skip.shape[-1], mode="linear", align_corners=False)
        projected = self.project(resized)
        return self.refine(torch.cat((projected, skip), dim=1))


class TemporalUNetGenerator(nn.Module):
    """One-dimensional residual U-Net for fixed or variable length windows.

    Input and output tensors have shape ``[batch, channel, time]``.  The final
    residual parameterization makes identity restoration easy to represent and
    bounds only the learned correction, not the input signal itself.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        super().__init__()
        self.config = config or GeneratorConfig()
        if self.config.depth < 1:
            raise ValueError("Generator depth must be at least one.")
        if self.config.in_channels != 1:
            raise ValueError("The demo currently supports univariate signals only.")

        channels = [
            min(self.config.base_channels * (2**level), self.config.max_channels)
            for level in range(self.config.depth + 1)
        ]
        self.stem = _ConvBlock(
            self.config.in_channels,
            channels[0],
            stride=1,
            groups=self.config.normalization_groups,
        )
        self.encoders = nn.ModuleList(
            _ConvBlock(
                channels[level],
                channels[level + 1],
                stride=2,
                groups=self.config.normalization_groups,
                dropout=self.config.dropout if level == self.config.depth - 1 else 0.0,
            )
            for level in range(self.config.depth)
        )

        skip_channels = list(reversed(channels[:-1]))
        decoder_in = channels[-1]
        decoder_blocks: list[nn.Module] = []
        for skip_channels_at_level in skip_channels:
            decoder_blocks.append(
                _DecoderBlock(
                    decoder_in,
                    skip_channels_at_level,
                    skip_channels_at_level,
                    groups=self.config.normalization_groups,
                    dropout=self.config.dropout,
                )
            )
            decoder_in = skip_channels_at_level
        self.decoders = nn.ModuleList(decoder_blocks)
        self.output = nn.Conv1d(channels[0], self.config.in_channels, kernel_size=1)

        # Starting near the identity makes the untrained checkpoint safe to
        # inspect and avoids large early corrections during training.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_signal_tensor(inputs)
        skips = [self.stem(inputs)]
        encoded = skips[0]
        for encoder in self.encoders:
            encoded = encoder(encoded)
            skips.append(encoded)

        decoded = skips[-1]
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            decoded = decoder(decoded, skip)

        correction = torch.tanh(self.output(decoded)) * self.config.residual_scale
        return inputs + correction


class TemporalPatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator over local temporal segments."""

    def __init__(self, config: DiscriminatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or DiscriminatorConfig()
        if self.config.layers < 1:
            raise ValueError("Discriminator layers must be at least one.")

        in_channels = self.config.signal_channels * 2
        blocks: list[nn.Module] = []
        for level in range(self.config.layers):
            out_channels = min(
                self.config.base_channels * (2**level), self.config.max_channels
            )
            blocks.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=2, padding=3)
            )
            if level:
                blocks.append(
                    nn.GroupNorm(
                        _normalization_groups(
                            out_channels, self.config.normalization_groups
                        ),
                        out_channels,
                    )
                )
            blocks.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        blocks.extend(
            [
                nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(in_channels, 1, kernel_size=3, padding=1),
            ]
        )
        self.net = nn.Sequential(*blocks)

    def forward(self, observation: Tensor, candidate: Tensor) -> Tensor:
        _validate_signal_tensor(observation)
        _validate_signal_tensor(candidate)
        if observation.shape != candidate.shape:
            raise ValueError("Observation and candidate tensors must have identical shapes.")
        return self.net(torch.cat((observation, candidate), dim=1))


class CausalForecasterScorer(nn.Module):
    """Compact causal forecaster and timestamp-level prediction-error scorer.

    For each time ``t``, the network receives only the preceding ``L`` samples.
    The initial ``L`` error scores are set to zero because a complete context is
    unavailable.  Calling :meth:`freeze` disables parameter gradients while
    preserving gradients with respect to a generator candidate.
    """

    def __init__(self, config: ForecasterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ForecasterConfig()
        if self.config.context_length < 2:
            raise ValueError("Forecaster context_length must be at least two.")
        layers: list[nn.Module] = [
            nn.Linear(self.config.context_length, self.config.hidden_channels),
            nn.SiLU(),
        ]
        if self.config.dropout > 0:
            layers.append(nn.Dropout(self.config.dropout))
        layers.extend(
            [
                nn.Linear(self.config.hidden_channels, self.config.hidden_channels),
                nn.SiLU(),
                nn.Linear(self.config.hidden_channels, 1),
            ]
        )
        self.predictor = nn.Sequential(*layers)

    def forward(self, signal: Tensor) -> Tensor:
        """Return one-step predictions with the same shape as ``signal``."""

        _validate_signal_tensor(signal)
        context = self.config.context_length
        padded = F.pad(signal, (context, 0), mode="replicate")
        windows = padded.unfold(dimension=-1, size=context, step=1)[..., : signal.shape[-1], :]
        flat_windows = windows.reshape(-1, context)
        predictions = self.predictor(flat_windows)
        return predictions.reshape_as(signal)

    def anomaly_score(self, signal: Tensor) -> Tensor:
        """Return squared causal prediction error at each timestamp."""

        predictions = self(signal)
        scores = (signal - predictions).square()
        scores = scores.clone()
        scores[..., : self.config.context_length] = 0.0
        return scores

    def freeze(self) -> "CausalForecasterScorer":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze(self) -> "CausalForecasterScorer":
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return self


def _validate_signal_tensor(signal: Tensor) -> None:
    if signal.ndim != 3:
        raise ValueError("Expected signal tensor with shape [batch, channel, time].")
    if signal.shape[1] != 1:
        raise ValueError("DenoiseAPT currently supports one signal channel.")
    if signal.shape[-1] < 8:
        raise ValueError("A signal window must contain at least eight timestamps.")
    if not signal.is_floating_point():
        raise TypeError("Signal tensors must use a floating-point dtype.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Return a model-size summary used by the training CLI and diagnostics."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
