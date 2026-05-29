from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.general import AlgebraistGeneralKernel
from stark.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.algebraist.allocator import AlgebraistAllocator
from stark.contracts.acceleration import AcceleratorLike
from stark.contracts.translations import Translation

try:
    from stark.algebraist.layout import AlgebraistLayout
except Exception:  # pragma: no cover
    AlgebraistLayout = object  # type: ignore[misc, assignment]

TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(slots=True)
class AlgebraistRuntimeGeneral(Generic[TranslationType]):
    """Runtime provider of general arity-based linear-combination kernels."""

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    layout: AlgebraistLayout | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)
    _support: AlgebraistRuntimeSupport[TranslationType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._support = AlgebraistRuntimeSupport(
            translation=self.translation,
            allocator=self.allocator,
            layout=self.layout,
            linear_combine=self.linear_combine,
            accelerator=self.accelerator,
        )

    def provide(self, request: AlgebraistArity) -> AlgebraistGeneralKernel[TranslationType]:
        return self._support.provide_general(request)

    def as_tuple(self, max_arity: int = 12) -> tuple[AlgebraistGeneralKernel[TranslationType], ...]:
        return self._support.provide_tuple(max_arity)


__all__ = ["AlgebraistRuntimeGeneral"]
