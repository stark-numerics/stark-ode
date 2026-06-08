from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class AlgebraistLayoutNormPolicy(Protocol):
    """Policy describing whether and how a layout field contributes to norms."""

    @property
    def include(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class AlgebraistLayoutNormRMS:
    """Use a root-mean-square field norm."""

    @property
    def include(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AlgebraistLayoutNormMax:
    """Use a maximum absolute-entry field norm."""

    @property
    def include(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AlgebraistLayoutNormExcluded:
    """Exclude this field from layout-aware norms."""

    @property
    def include(self) -> bool:
        return False


__all__ = [
    "AlgebraistLayoutNormExcluded",
    "AlgebraistLayoutNormMax",
    "AlgebraistLayoutNormPolicy",
    "AlgebraistLayoutNormRMS",
]
