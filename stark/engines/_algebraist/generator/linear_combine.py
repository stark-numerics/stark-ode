from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.core.contracts.frame import FrameLike
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines._algebraist.arity import AlgebraistArity
from stark.engines._algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines._algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines._algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetMutable,
)
from stark.engines._algebraist.allocator import AlgebraistAllocator
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.translation import Translation

TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorLinearCombine(Generic[TranslationType]):
    """Generated provider of general arity-based linear-combination kernels."""

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    frame: FrameLike
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    target: AlgebraistGeneratorTarget = field(default_factory=AlgebraistGeneratorTargetMutable)

    def source_string(self, request: AlgebraistArity) -> str:
        source = getattr(self.target, "source_linear_combine", None)
        if callable(source):
            return cast(str, source(self.frame, request))
        return AlgebraistGeneratorEmitter(self.frame, target=self.target).general(request)

    def compile(self, source: str) -> Callable[..., TranslationType]:
        return cast(
            Callable[..., TranslationType],
            AlgebraistGeneratorCompiler(self.accelerator).compile(source),
        )

    def provide(self, request: AlgebraistArity) -> Callable[..., TranslationType]:
        return self.compile(self.source_string(request))
