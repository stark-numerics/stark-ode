from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.accelerators.none import AcceleratorNone
from stark.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.algebraist.layout import AlgebraistLayout
from stark.algebraist.stencil import AlgebraistStencil
from stark.algebraist.allocator import AlgebraistAllocator
from stark.contracts.accelerator import Accelerator

StateType = TypeVar("StateType")
TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorSpecialist(Generic[StateType, TranslationType]):
    """Generated provider of fixed-coefficient scheme kernels.

    ``stencil.apply`` selects whether the provided kernel writes a translation
    delta or applies that delta to an origin state.
    """

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    layout: AlgebraistLayout
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def source_string(self, stencil: AlgebraistStencil) -> str:
        return AlgebraistGeneratorEmitter(self.layout).specialist(stencil)

    def source_unit_apply(self) -> str:
        return AlgebraistGeneratorEmitter(self.layout).unit_apply()

    def compile(self, source: str) -> Callable[..., object]:
        return cast(
            Callable[..., object],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, stencil: AlgebraistStencil) -> Callable[..., object]:
        return self.compile(self.source_string(stencil))

    def provide_unit_apply(self) -> Callable[..., object]:
        return self.compile(self.source_unit_apply())


__all__ = ["AlgebraistGeneratorSpecialist"]
