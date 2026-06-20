from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetMutable:
    """Emit kernels that update mutable output arrays in place."""


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetMutableVectorized:
    """Emit mutable whole-array assignments for backend array fields."""


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetFunctional:
    """Emit kernels that return updated arrays instead of mutating them."""


AlgebraistGeneratorTarget: TypeAlias = (
    AlgebraistGeneratorTargetMutable
    | AlgebraistGeneratorTargetMutableVectorized
    | AlgebraistGeneratorTargetFunctional
)


__all__ = [
    "AlgebraistGeneratorTarget",
    "AlgebraistGeneratorTargetFunctional",
    "AlgebraistGeneratorTargetMutable",
    "AlgebraistGeneratorTargetMutableVectorized",
]
