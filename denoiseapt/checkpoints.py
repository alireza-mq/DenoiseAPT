"""Versioned checkpoint save/load utilities for the demo models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .concern import ConcernConfig
from .models import (
    CausalForecasterScorer,
    ForecasterConfig,
    GeneratorConfig,
    TemporalUNetGenerator,
)


CHECKPOINT_FORMAT_VERSION = 1


@dataclass
class ModelBundle:
    """Inference components and metadata restored from one checkpoint."""

    generator: TemporalUNetGenerator
    scorer: CausalForecasterScorer
    baseline_generator: TemporalUNetGenerator | None
    concern_config: ConcernConfig
    metadata: dict[str, Any]
    path: Path | None = None

    def eval(self) -> "ModelBundle":
        self.generator.eval()
        self.scorer.freeze()
        if self.baseline_generator is not None:
            self.baseline_generator.eval()
        return self


def save_model_bundle(
    path: str | Path,
    *,
    generator: TemporalUNetGenerator,
    scorer: CausalForecasterScorer,
    baseline_generator: TemporalUNetGenerator | None = None,
    concern_config: ConcernConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save all inference components in one portable CPU bundle."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "generator_config": generator.config.to_dict(),
        "generator_state": _cpu_state_dict(generator),
        "scorer_config": scorer.config.to_dict(),
        "scorer_state": _cpu_state_dict(scorer),
        "concern_config": (concern_config or ConcernConfig()).to_dict(),
        "metadata": dict(metadata or {}),
    }
    if baseline_generator is not None:
        payload.update(
            {
                "baseline_generator_config": baseline_generator.config.to_dict(),
                "baseline_generator_state": _cpu_state_dict(baseline_generator),
            }
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_model_bundle(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> ModelBundle:
    """Load a DenoiseAPT bundle while rejecting incompatible future formats."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {source}")
    map_location = torch.device(device)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:
        # Compatibility path for old PyTorch; use it only with trusted bundles.
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    version = int(payload.get("format_version", -1))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {version}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    generator = TemporalUNetGenerator(
        GeneratorConfig.from_dict(_mapping(payload, "generator_config"))
    ).to(map_location)
    generator.load_state_dict(_mapping(payload, "generator_state"), strict=strict)
    scorer = CausalForecasterScorer(
        ForecasterConfig.from_dict(_mapping(payload, "scorer_config"))
    ).to(map_location)
    scorer.load_state_dict(_mapping(payload, "scorer_state"), strict=strict)

    baseline: TemporalUNetGenerator | None = None
    if "baseline_generator_state" in payload:
        baseline = TemporalUNetGenerator(
            GeneratorConfig.from_dict(_mapping(payload, "baseline_generator_config"))
        ).to(map_location)
        baseline.load_state_dict(
            _mapping(payload, "baseline_generator_state"), strict=strict
        )

    bundle = ModelBundle(
        generator=generator,
        scorer=scorer,
        baseline_generator=baseline,
        concern_config=ConcernConfig.from_dict(_mapping(payload, "concern_config")),
        metadata=dict(_mapping(payload, "metadata")),
        path=source,
    )
    return bundle.eval()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint field {key!r} is missing or invalid.")
    return value
