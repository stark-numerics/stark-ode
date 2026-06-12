from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from stark.engines.algebraist.layout import (
    AlgebraistLayoutNormExcluded,
    AlgebraistLayoutNormMax,
    AlgebraistLayoutNormPolicy,
    AlgebraistLayoutNormRMS,
)


class LayoutNormPolicy(Protocol):
    """User-facing norm policy for one `LayoutField`."""

    @property
    def include(self) -> bool: ...

    def to_algebraist_norm(self) -> AlgebraistLayoutNormPolicy: ...


@dataclass(frozen=True, slots=True)
class LayoutNormRMS:
    """Use a root-mean-square field norm."""

    @property
    def include(self) -> bool:
        return True

    def to_algebraist_norm(self) -> AlgebraistLayoutNormPolicy:
        return AlgebraistLayoutNormRMS()


@dataclass(frozen=True, slots=True)
class LayoutNormMax:
    """Use a maximum absolute-entry field norm."""

    @property
    def include(self) -> bool:
        return True

    def to_algebraist_norm(self) -> AlgebraistLayoutNormPolicy:
        return AlgebraistLayoutNormMax()


@dataclass(frozen=True, slots=True)
class LayoutNormExcluded:
    """Exclude this field from layout-aware norms."""

    @property
    def include(self) -> bool:
        return False

    def to_algebraist_norm(self) -> AlgebraistLayoutNormPolicy:
        return AlgebraistLayoutNormExcluded()


__all__ = [
    "LayoutNormExcluded",
    "LayoutNormMax",
    "LayoutNormPolicy",
    "LayoutNormRMS",
]
