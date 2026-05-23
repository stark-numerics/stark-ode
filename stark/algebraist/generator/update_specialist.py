from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.algebraist.layout import AlgebraistLayout
from stark.algebraist.update_specialist import AlgebraistUpdateKernel, AlgebraistUpdateStencil
from stark.algebraist.workbench import AlgebraistWorkbench
from stark.contracts.acceleration import AcceleratorLike

StateType = TypeVar("StateType")
TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorUpdateSpecialist(Generic[StateType, TranslationType]):
    """Generated provider of fixed-coefficient state-update kernels."""

    translation: TranslationType
    workbench: AlgebraistWorkbench[TranslationType]
    layout: AlgebraistLayout
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)

    def source_string(self, stencil: AlgebraistUpdateStencil) -> str:
        return AlgebraistGeneratorEmitter(self.layout).update(stencil)

    def compile(self, source: str) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        return cast(
            AlgebraistUpdateKernel[StateType, TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, stencil: AlgebraistUpdateStencil) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        return self.compile(self.source_string(stencil))
