from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines.algebraist.frame import AlgebraistFrame
from stark.engines.algebraist.allocator import AlgebraistAllocator
from stark.core.contracts.accelerator import Accelerator

TranslationType = TypeVar("TranslationType")


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorLinearCombine(Generic[TranslationType]):
    """Generated provider of general arity-based linear-combination kernels."""

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    frame: AlgebraistFrame
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def source_string(self, request: AlgebraistArity) -> str:
        return AlgebraistGeneratorEmitter(self.frame).general(request)

    def compile(self, source: str) -> Callable[..., TranslationType]:
        return cast(
            Callable[..., TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: AlgebraistArity) -> Callable[..., TranslationType]:
        return self.compile(self.source_string(request))
