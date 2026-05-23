from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.algebraist.layout import AlgebraistLayout
from stark.algebraist.workbench import AlgebraistWorkbench
from stark.contracts.acceleration import AcceleratorLike

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorGeneral(Generic[TranslationType]):
    """Generated provider of general arity-based linear-combination kernels."""

    translation: TranslationType
    workbench: AlgebraistWorkbench[TranslationType]
    layout: AlgebraistLayout
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)

    def source_string(self, request: AlgebraistArity) -> str:
        return AlgebraistGeneratorEmitter(self.layout).general(request)

    def compile(self, source: str) -> Callable[..., TranslationType]:
        return cast(
            Callable[..., TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: AlgebraistArity) -> Callable[..., TranslationType]:
        return self.compile(self.source_string(request))
