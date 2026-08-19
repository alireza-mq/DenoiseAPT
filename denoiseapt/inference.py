"""Deterministic CPU inference and stochastic disagreement for DenoiseAPT."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time
from typing import Any, Iterator, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray
import torch
from torch import nn

from .actions import SignalSession
from .checkpoints import ModelBundle, load_model_bundle
from .concern import ConcernConfig, ConcernCues, compute_concern_cues
from .models import CausalForecasterScorer, TemporalUNetGenerator


@dataclass(frozen=True)
class InferenceConfig:
    """Reproducibility and Monte Carlo settings for inference."""

    stochastic_passes: int = 12
    seed: int = 17
    deterministic_algorithms: bool = True
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.stochastic_passes < 2:
            raise ValueError("At least two stochastic passes are required for disagreement.")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "InferenceConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindowNormalization:
    """Robust affine transform fitted once per observation window."""

    center: float
    scale: float

    @classmethod
    def fit(cls, signal: NDArray[np.float32]) -> "WindowNormalization":
        center = float(np.median(signal))
        q25, q75 = np.percentile(signal, [25.0, 75.0])
        robust_scale = float((q75 - q25) / 1.349)
        scale = max(robust_scale, float(np.std(signal)), 1e-4)
        return cls(center=center, scale=scale)

    def normalize(self, signal: ArrayLike) -> NDArray[np.float32]:
        return ((np.asarray(signal, dtype=np.float32) - self.center) / self.scale).astype(
            np.float32
        )

    def restore(self, signal: ArrayLike) -> NDArray[np.float32]:
        return (np.asarray(signal, dtype=np.float32) * self.scale + self.center).astype(
            np.float32
        )


@dataclass(frozen=True)
class DenoiseResult:
    """Signals, scores, concern cues, and provenance from one inference run."""

    observation: NDArray[np.float32]
    candidate: NDArray[np.float32]
    stochastic_candidates: NDArray[np.float32]
    observation_scores: NDArray[np.float32]
    candidate_scores: NDArray[np.float32]
    concern: ConcernCues
    baseline_candidate: NDArray[np.float32] | None
    baseline_scores: NDArray[np.float32] | None
    normalization: WindowNormalization
    latency_ms: float
    stochastic_passes: int
    seed: int

    def new_session(self) -> SignalSession:
        return SignalSession(self.observation, self.candidate)

    def to_dict(self, *, include_stochastic_candidates: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "observation": self.observation.tolist(),
            "candidate": self.candidate.tolist(),
            "observation_scores": self.observation_scores.tolist(),
            "candidate_scores": self.candidate_scores.tolist(),
            "concern": self.concern.to_dict(),
            "baseline_candidate": (
                None if self.baseline_candidate is None else self.baseline_candidate.tolist()
            ),
            "baseline_scores": (
                None if self.baseline_scores is None else self.baseline_scores.tolist()
            ),
            "normalization": asdict(self.normalization),
            "latency_ms": float(self.latency_ms),
            "stochastic_passes": self.stochastic_passes,
            "seed": self.seed,
        }
        if include_stochastic_candidates:
            data["stochastic_candidates"] = self.stochastic_candidates.tolist()
        return data


class DenoiseAPTPipeline:
    """Inference facade shared by the API server and offline smoke tests."""

    def __init__(
        self,
        generator: TemporalUNetGenerator,
        scorer: CausalForecasterScorer,
        *,
        baseline_generator: TemporalUNetGenerator | None = None,
        concern_config: ConcernConfig | None = None,
        inference_config: InferenceConfig | None = None,
        device: str | torch.device = "cpu",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.generator = generator.to(self.device).eval()
        self.scorer = scorer.to(self.device).freeze()
        self.baseline_generator = (
            None if baseline_generator is None else baseline_generator.to(self.device).eval()
        )
        self.concern_config = concern_config or ConcernConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.metadata = dict(metadata or {})
        set_deterministic(
            self.inference_config.seed,
            deterministic_algorithms=self.inference_config.deterministic_algorithms,
            torch_threads=self.inference_config.torch_threads,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        inference_config: InferenceConfig | None = None,
    ) -> "DenoiseAPTPipeline":
        bundle = load_model_bundle(path, device=device)
        return cls.from_bundle(
            bundle, device=device, inference_config=inference_config
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: ModelBundle,
        *,
        device: str | torch.device = "cpu",
        inference_config: InferenceConfig | None = None,
    ) -> "DenoiseAPTPipeline":
        return cls(
            bundle.generator,
            bundle.scorer,
            baseline_generator=bundle.baseline_generator,
            concern_config=bundle.concern_config,
            inference_config=inference_config,
            device=device,
            metadata=bundle.metadata,
        )

    def run(
        self,
        observation: ArrayLike,
        *,
        stochastic_passes: int | None = None,
        seed: int | None = None,
    ) -> DenoiseResult:
        """Denoise one window and compute its local preservation-concern map."""

        values = _signal(observation)
        normalization = WindowNormalization.fit(values)
        normalized = normalization.normalize(values)
        input_tensor = torch.from_numpy(normalized).to(self.device)[None, None, :]
        passes = int(stochastic_passes or self.inference_config.stochastic_passes)
        if passes < 2:
            raise ValueError("At least two stochastic passes are required.")
        inference_seed = self.inference_config.seed if seed is None else int(seed)

        started = time.perf_counter()
        candidates_normalized = self._stochastic_forward(
            input_tensor, passes=passes, seed=inference_seed
        )
        aggregate_normalized = np.median(candidates_normalized, axis=0).astype(np.float32)
        observation_scores = self.score_normalized(normalized)
        candidate_scores = self.score_normalized(aggregate_normalized)

        baseline_normalized: NDArray[np.float32] | None = None
        baseline_scores: NDArray[np.float32] | None = None
        if self.baseline_generator is not None:
            baseline_normalized = self._deterministic_forward(
                self.baseline_generator, input_tensor
            )
            baseline_scores = self.score_normalized(baseline_normalized)

        concern = compute_concern_cues(
            normalized,
            aggregate_normalized,
            candidates_normalized,
            observation_scores,
            candidate_scores,
            self.concern_config,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        candidates = np.stack(
            [normalization.restore(candidate) for candidate in candidates_normalized]
        ).astype(np.float32)
        baseline_candidate = (
            None
            if baseline_normalized is None
            else normalization.restore(baseline_normalized)
        )
        return DenoiseResult(
            observation=values,
            candidate=normalization.restore(aggregate_normalized),
            stochastic_candidates=candidates,
            observation_scores=observation_scores,
            candidate_scores=candidate_scores,
            concern=concern,
            baseline_candidate=baseline_candidate,
            baseline_scores=baseline_scores,
            normalization=normalization,
            latency_ms=latency_ms,
            stochastic_passes=passes,
            seed=inference_seed,
        )

    def score_signal(
        self, signal: ArrayLike, normalization: WindowNormalization | None = None
    ) -> NDArray[np.float32]:
        values = _signal(signal)
        transform = normalization or WindowNormalization.fit(values)
        return self.score_normalized(transform.normalize(values))

    def score_normalized(self, normalized_signal: ArrayLike) -> NDArray[np.float32]:
        values = _signal(normalized_signal)
        tensor = torch.from_numpy(values).to(self.device)[None, None, :]
        with torch.inference_mode():
            scores = self.scorer.anomaly_score(tensor)
        return scores[0, 0].detach().cpu().numpy().astype(np.float32)

    def _stochastic_forward(
        self, inputs: torch.Tensor, *, passes: int, seed: int
    ) -> NDArray[np.float32]:
        candidates: list[NDArray[np.float32]] = []
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), _mc_dropout(self.generator):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                for _ in range(passes):
                    output = self.generator(inputs)
                    candidates.append(
                        output[0, 0].detach().cpu().numpy().astype(np.float32)
                    )
        return np.stack(candidates)

    @staticmethod
    def _deterministic_forward(
        model: TemporalUNetGenerator, inputs: torch.Tensor
    ) -> NDArray[np.float32]:
        model.eval()
        with torch.inference_mode():
            output = model(inputs)
        return output[0, 0].detach().cpu().numpy().astype(np.float32)


@contextmanager
def _mc_dropout(model: nn.Module) -> Iterator[None]:
    """Enable only dropout modules, retaining evaluation behavior elsewhere."""

    training_states = {module: module.training for module in model.modules()}
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
    try:
        yield
    finally:
        for module, training in training_states.items():
            module.train(training)


def set_deterministic(
    seed: int,
    *,
    deterministic_algorithms: bool = True,
    torch_threads: int = 1,
) -> None:
    """Set reproducible random state for training and CPU demo inference."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(max(1, int(torch_threads)))
    torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)


def _signal(values: ArrayLike) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size < 8 or not np.all(np.isfinite(array)):
        raise ValueError("Expected a one-dimensional signal with at least eight finite values.")
    return array
