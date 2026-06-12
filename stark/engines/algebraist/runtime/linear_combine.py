from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.linear_combine import AlgebraistLinearCombineKernel
from stark.engines.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.engines.algebraist.allocator import AlgebraistAllocator
from stark.contracts.accelerator import Accelerator
from stark.contracts.translation import Translation

try:
    from stark.engines.algebraist.layout import AlgebraistLayout
except Exception:  # pragma: no cover
    AlgebraistLayout = object  # type: ignore[misc, assignment]

TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(slots=True)
class AlgebraistRuntimeLinearCombine(Generic[TranslationType]):
    """Runtime provider of general arity-based linear-combination kernels."""

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    layout: AlgebraistLayout | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    _support: AlgebraistRuntimeSupport[TranslationType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._support = AlgebraistRuntimeSupport(
            translation=self.translation,
            allocator=self.allocator,
            layout=self.layout,
            linear_combine=self.linear_combine,
            accelerator=self.accelerator,
        )

    def provide(self, request: AlgebraistArity) -> AlgebraistLinearCombineKernel[TranslationType]:
        return self._support.provide_linear_combine(request)

    def as_tuple(self, max_arity: int = 12) -> tuple[AlgebraistLinearCombineKernel[TranslationType], ...]:
        return self._support.provide_tuple(max_arity)


__all__ = ["AlgebraistRuntimeLinearCombine"]
