from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.core.contracts.frame import FrameLike
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.engines.algebraist.stencil import AlgebraistStencil
from stark.engines.algebraist.allocator import AlgebraistAllocator
from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.state import State
from stark.core.contracts.translation import Translation

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(slots=True)
class AlgebraistRuntimeSpecialist(Generic[StateType, TranslationType]):
    """Runtime provider of fixed-coefficient scheme kernels.

    ``provide_delta`` writes a translation delta. ``provide_apply`` applies
    that delta to an origin state. The split keeps call sites explicit.
    """

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    frame: FrameLike | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    _support: AlgebraistRuntimeSupport[TranslationType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._support = AlgebraistRuntimeSupport(
            translation=self.translation,
            allocator=self.allocator,
            frame=self.frame,
            linear_combine=self.linear_combine,
            accelerator=self.accelerator,
        )

    def provide_delta(self, stencil: AlgebraistStencil) -> Callable[..., TranslationType]:
        return self._support.provide_delta(request=stencil)

    def provide_apply(self, stencil: AlgebraistStencil) -> Callable[..., StateType]:
        return self._support.provide_apply(request=stencil)

    def provide_unit_apply(self) -> Callable[..., StateType]:
        return cast(Callable[..., StateType], self._support.provide_unit_apply())


__all__ = ["AlgebraistRuntimeSpecialist"]
