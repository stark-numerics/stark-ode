from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.frame import AlgebraistFrame
from stark.engines.shared.algebraist.stencil import AlgebraistStencil


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetMutable:
    """Emit kernels that update mutable output arrays in place."""


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetMutableVectorized:
    """Emit mutable whole-array assignments for backend array fields."""


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorTargetFunctional:
    """Emit kernels that return updated arrays instead of mutating them."""


class AlgebraistGeneratorTargetCustom(Protocol):
    """Backend-owned target that provides generated source directly."""

    def source_linear_combine(self, frame: AlgebraistFrame, request: AlgebraistArity) -> str: ...
    def source_specialist(self, frame: AlgebraistFrame, stencil: AlgebraistStencil) -> str: ...
    def source_unit_apply(self, frame: AlgebraistFrame) -> str: ...
    def source_norm(self, frame: AlgebraistFrame) -> str: ...
    def source_inner_product(self, frame: AlgebraistFrame) -> str: ...


AlgebraistGeneratorTarget: TypeAlias = (
    AlgebraistGeneratorTargetMutable
    | AlgebraistGeneratorTargetMutableVectorized
    | AlgebraistGeneratorTargetFunctional
    | AlgebraistGeneratorTargetCustom
)


__all__ = [
    "AlgebraistGeneratorTarget",
    "AlgebraistGeneratorTargetCustom",
    "AlgebraistGeneratorTargetFunctional",
    "AlgebraistGeneratorTargetMutable",
    "AlgebraistGeneratorTargetMutableVectorized",
]
