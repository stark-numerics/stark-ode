from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

MAX_UNRAVEL_SIZE = 16


def _normalize_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    normalized = tuple(shape)
    if not normalized:
        raise ValueError("shape must contain at least one dimension.")
    for dimension in normalized:
        if not isinstance(dimension, int):
            raise TypeError("shape dimensions must be integers.")
        if dimension <= 0:
            raise ValueError("shape dimensions must be positive.")
    return normalized


class AlgebraistFramePolicy(Protocol):
    """Policy describing how a frame field is emitted/traversed."""


@dataclass(frozen=True, slots=True)
class AlgebraistFrameScalar:
    """Emit direct scalar assignment with no indexing or broadcasting."""


@dataclass(frozen=True, slots=True)
class AlgebraistFrameBroadcast:
    """Emit whole-field broadcast assignment, for example target[...] = expression."""


@dataclass(frozen=True, slots=True)
class AlgebraistFrameLooped:
    """Emit generated loops over a known shape or runtime rank."""

    rank: int | None = None
    shape: tuple[int, ...] | list[int] | None = None

    def __post_init__(self) -> None:
        shape = self.shape
        rank = self.rank

        if shape is not None:
            normalized = _normalize_shape(shape)
            object.__setattr__(self, "shape", normalized)
            if rank is not None and rank != len(normalized):
                raise ValueError("rank must match len(shape).")
            object.__setattr__(self, "rank", len(normalized))
            return

        if rank is None:
            raise ValueError("AlgebraistFrameLooped requires rank or shape.")
        if not isinstance(rank, int):
            raise TypeError("rank must be an integer.")
        if rank <= 0:
            raise ValueError("rank must be positive.")


@dataclass(frozen=True, slots=True)
class AlgebraistFrameUnravel:
    """Emit explicit per-index assignments for a known finite shape.

    This policy is a hard request: generated code must be unravelled. If the
    shape is too large for safe source emission, construction fails rather than
    silently falling back to looped emission.
    """

    shape: tuple[int, ...] | list[int]

    def __post_init__(self) -> None:
        normalized = _normalize_shape(self.shape)
        size = prod(normalized)
        if size > MAX_UNRAVEL_SIZE:
            raise ValueError(
                "AlgebraistFrameUnravel requires explicit unravelled emission, "
                f"but shape {normalized!r} has {size} entries; use "
                "AlgebraistFrameLooped explicitly for looped emission."
            )
        object.__setattr__(self, "shape", normalized)
