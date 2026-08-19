"""Reversible interval interventions for the interactive workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


ActionName = Literal["accept", "protect", "blend", "revert"]


@dataclass(frozen=True)
class ActionRecord:
    """Immutable audit entry for one half-open interval edit."""

    revision: int
    action: ActionName
    start: int
    end: int
    beta: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_interval_action(
    current: ArrayLike,
    observation: ArrayLike,
    candidate: ArrayLike,
    *,
    action: Literal["accept", "protect", "blend"],
    start: int,
    end: int,
    beta: float = 0.5,
) -> NDArray[np.float32]:
    """Return a copy with one configured interval action applied.

    Intervals use Python's half-open convention ``[start, end)``.  For Blend,
    ``beta=1`` keeps the observation and ``beta=0`` keeps the candidate.
    """

    approved = _signal(current, "current")
    observed = _signal(observation, "observation")
    restored = _signal(candidate, "candidate")
    if approved.size != observed.size or approved.size != restored.size:
        raise ValueError("Current, observation, and candidate lengths must match.")
    start_index, end_index = _validate_interval(start, end, approved.size)
    result = approved.copy()
    if action == "accept":
        result[start_index:end_index] = restored[start_index:end_index]
    elif action == "protect":
        result[start_index:end_index] = observed[start_index:end_index]
    elif action == "blend":
        blend_weight = float(beta)
        if not np.isfinite(blend_weight) or not 0.0 <= blend_weight <= 1.0:
            raise ValueError("Blend beta must be a finite value in [0, 1].")
        result[start_index:end_index] = (
            blend_weight * observed[start_index:end_index]
            + (1.0 - blend_weight) * restored[start_index:end_index]
        )
    else:
        raise ValueError(f"Unsupported interval action: {action!r}")
    return result.astype(np.float32)


class SignalSession:
    """Maintain an approved signal and a deterministic reversible edit history."""

    def __init__(self, observation: ArrayLike, candidate: ArrayLike) -> None:
        self.observation = _signal(observation, "observation")
        self.candidate = _signal(candidate, "candidate")
        if self.observation.size != self.candidate.size:
            raise ValueError("Observation and candidate lengths must match.")
        self.current = self.candidate.copy()
        # Each snapshot is the complete pre-edit state, so reversion is exact.
        self._states: list[NDArray[np.float32]] = []
        self._records: list[ActionRecord] = []
        self._revision = 0

    @property
    def records(self) -> tuple[ActionRecord, ...]:
        return tuple(self._records)

    @property
    def can_revert(self) -> bool:
        return bool(self._states)

    def apply(
        self,
        action: ActionName,
        start: int | None = None,
        end: int | None = None,
        *,
        beta: float = 0.5,
    ) -> NDArray[np.float32]:
        if action == "revert":
            return self.revert()
        start_index = 0 if start is None else int(start)
        end_index = self.current.size if end is None else int(end)
        start_index, end_index = _validate_interval(
            start_index, end_index, self.current.size
        )
        self._states.append(self.current.copy())
        self.current = apply_interval_action(
            self.current,
            self.observation,
            self.candidate,
            action=action,
            start=start_index,
            end=end_index,
            beta=beta,
        )
        self._revision += 1
        self._records.append(
            ActionRecord(
                revision=self._revision,
                action=action,
                start=start_index,
                end=end_index,
                beta=float(beta) if action == "blend" else None,
            )
        )
        return self.current.copy()

    def revert(self) -> NDArray[np.float32]:
        if not self._states:
            raise RuntimeError("No prior intervention is available to revert.")
        self.current = self._states.pop()
        self._revision += 1
        self._records.append(
            ActionRecord(
                revision=self._revision,
                action="revert",
                start=0,
                end=self.current.size,
                beta=None,
            )
        )
        return self.current.copy()

    def reset(self) -> NDArray[np.float32]:
        """Start a fresh action history from the model candidate."""

        self.current = self.candidate.copy()
        self._states.clear()
        self._records.clear()
        self._revision = 0
        return self.current.copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.tolist(),
            "candidate": self.candidate.tolist(),
            "approved": self.current.tolist(),
            # Compatibility aliases used by the deliberately small HTTP layer.
            "current": self.current.tolist(),
            "can_revert": self.can_revert,
            "history_depth": len(self._states),
            "revision": self._revision,
            "actions": [record.to_dict() for record in self._records],
        }


def _validate_interval(start: int, end: int, length: int) -> tuple[int, int]:
    start_index = int(start)
    end_index = int(end)
    if start_index < 0 or end_index > length or start_index >= end_index:
        raise ValueError(
            f"Invalid half-open interval [{start_index}, {end_index}) for length {length}."
        )
    return start_index, end_index


def _signal(values: ArrayLike, name: str) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty one-dimensional finite signal.")
    return array.copy()
