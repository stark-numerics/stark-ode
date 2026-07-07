from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.core.contracts.frame import FrameLike
from stark.core.contracts.state import State
from stark.core.contracts.translation import Translation
from stark.engines.shared.accelerators.none import AcceleratorNone
from stark.engines.shared.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.shared.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines.shared.algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetMutable,
)
from stark.engines.shared.algebraist.stencil import AlgebraistStencil
from stark.engines.shared.algebraist.allocator import AlgebraistAllocator
from stark.core.contracts.accelerator import Accelerator

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorSpecialist(Generic[StateType, TranslationType]):
    """Generated provider of fixed-coefficient scheme kernels.

    ``provide_delta`` writes a translation delta. ``provide_apply`` applies
    that delta to an origin state. The split keeps call sites explicit.
    """

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    frame: FrameLike
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def source_string(self, stencil: AlgebraistStencil) -> str:
        source = getattr(self.target, "source_specialist", None)
        if callable(source):
            return cast(str, source(self.frame, stencil))
        return AlgebraistGeneratorEmitter(self.frame, target=self.target).specialist(stencil)

    def source_unit_apply(self) -> str:
        source = getattr(self.target, "source_unit_apply", None)
        if callable(source):
            return cast(str, source(self.frame))
        return AlgebraistGeneratorEmitter(self.frame, target=self.target).unit_apply()

    def compile(self, source: str) -> Callable[..., object]:
        return cast(
            Callable[..., object],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide_delta(self, stencil: AlgebraistStencil) -> Callable[..., TranslationType]:
        return cast(Callable[..., TranslationType], self.compile(self.source_string(stencil)))

    def provide_apply(self, stencil: AlgebraistStencil) -> Callable[..., StateType]:
        return cast(Callable[..., StateType], self.compile(self.source_string(stencil)))

    def provide_unit_apply(self) -> Callable[..., object]:
        return self.compile(self.source_unit_apply())


__all__ = ["AlgebraistGeneratorSpecialist"]
