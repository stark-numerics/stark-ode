from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.engines.shared.accelerators.none import AcceleratorNone
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.shared.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines.shared.algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetMutable,
)
from stark.engines.shared.algebraist.frame import AlgebraistFrame
from stark.engines.shared.algebraist.allocator import AlgebraistAllocator
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
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def source_string(self, request: AlgebraistArity) -> str:
        source = getattr(self.target, "source_linear_combine", None)
        if callable(source):
            return source(self.frame, request)
        return AlgebraistGeneratorEmitter(self.frame, target=self.target).general(request)

    def compile(self, source: str) -> Callable[..., TranslationType]:
        return cast(
            Callable[..., TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: AlgebraistArity) -> Callable[..., TranslationType]:
        return self.compile(self.source_string(request))
