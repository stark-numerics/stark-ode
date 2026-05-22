from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Protocol


class AlgebraistPolicy(Protocol):
    def normalized(self) -> "AlgebraistPolicy":
        ...


@dataclass(frozen=True, slots=True)
class AlgebraistBroadcast:
    def normalized(self) -> "AlgebraistBroadcast":
        return self


@dataclass(frozen=True, slots=True)
class AlgebraistLooped:
    rank: int | None = None
    shape: tuple[int, ...] | None = None

    def normalized(self) -> "AlgebraistLooped":
        shape = tuple(self.shape) if self.shape is not None else None
        rank = self.rank

        if shape is not None:
            validate_shape(shape)
            if rank is None:
                rank = len(shape)
            elif rank != len(shape):
                raise ValueError(f"Looped rank {rank} does not match shape {shape!r}.")

        if rank is None:
            raise ValueError("Looped Algebraist fields need an explicit rank or shape.")

        return replace(self, rank=rank, shape=shape)


@dataclass(frozen=True, slots=True)
class AlgebraistSmallFixed:
    shape: tuple[int, ...]

    def normalized(self) -> "AlgebraistSmallFixed":
        shape = tuple(self.shape)
        validate_shape(shape)
        if prod(shape) > 16:
            raise ValueError(f"Small-fixed Algebraist field is too large for unrolled code: {shape!r}.")
        return replace(self, shape=shape)


def validate_shape(shape: tuple[int, ...]) -> None:
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Invalid Algebraist field shape {shape!r}.")
