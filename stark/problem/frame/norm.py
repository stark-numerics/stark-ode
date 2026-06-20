from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stark.engines.shared.algebraist.frame import (
    AlgebraistFrameNormExcluded,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormPolicy,
    AlgebraistFrameNormRMS,
)


class FrameNormPolicy(Protocol):
    """User-facing norm policy for one `FrameField`."""

    @property
    def include(self) -> bool: ...

    def to_algebraist_norm(self) -> AlgebraistFrameNormPolicy: ...


@dataclass(frozen=True, slots=True)
class FrameNormRMS:
    """Use a root-mean-square field norm."""

    @property
    def include(self) -> bool:
        return True

    def to_algebraist_norm(self) -> AlgebraistFrameNormPolicy:
        return AlgebraistFrameNormRMS()


@dataclass(frozen=True, slots=True)
class FrameNormMax:
    """Use a maximum absolute-entry field norm."""

    @property
    def include(self) -> bool:
        return True

    def to_algebraist_norm(self) -> AlgebraistFrameNormPolicy:
        return AlgebraistFrameNormMax()


@dataclass(frozen=True, slots=True)
class FrameNormExcluded:
    """Exclude this field from frame-aware norms."""

    @property
    def include(self) -> bool:
        return False

    def to_algebraist_norm(self) -> AlgebraistFrameNormPolicy:
        return AlgebraistFrameNormExcluded()


__all__ = [
    "FrameNormExcluded",
    "FrameNormMax",
    "FrameNormPolicy",
    "FrameNormRMS",
]
