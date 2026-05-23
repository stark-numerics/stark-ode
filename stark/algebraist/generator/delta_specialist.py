from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.delta_specialist import AlgebraistDeltaKernel, AlgebraistDeltaStencil
from stark.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.algebraist.layout import AlgebraistLayout
from stark.algebraist.workbench import AlgebraistWorkbench
from stark.contracts.acceleration import AcceleratorLike

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorDeltaSpecialist(Generic[TranslationType]):
    """Generated provider of fixed-coefficient delta kernels."""

    translation: TranslationType
    workbench: AlgebraistWorkbench[TranslationType]
    layout: AlgebraistLayout
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)

    def source_string(self, stencil: AlgebraistDeltaStencil) -> str:
        return AlgebraistGeneratorEmitter(self.layout).delta(stencil)

    def compile(self, source: str) -> AlgebraistDeltaKernel[TranslationType]:
        return cast(
            AlgebraistDeltaKernel[TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, stencil: AlgebraistDeltaStencil) -> AlgebraistDeltaKernel[TranslationType]:
        return self.compile(self.source_string(stencil))
