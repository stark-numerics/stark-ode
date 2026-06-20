from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AlgebraistFrameNormPolicy(Protocol):
    """Policy describing whether and how a frame field contributes to norms."""

    @property
    def include(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AlgebraistFrameNormRMS:
    """Use a root-mean-square field norm."""

    @property
    def include(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AlgebraistFrameNormMax:
    """Use a maximum absolute-entry field norm."""

    @property
    def include(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AlgebraistFrameNormExcluded:
    """Exclude this field from frame-aware norms."""

    @property
    def include(self) -> bool:
        return False


__all__ = [
    "AlgebraistFrameNormExcluded",
    "AlgebraistFrameNormMax",
    "AlgebraistFrameNormPolicy",
    "AlgebraistFrameNormRMS",
]
