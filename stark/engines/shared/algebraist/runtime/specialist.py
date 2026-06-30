from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stark.engines.shared.accelerators.none import AcceleratorNone
from stark.engines.shared.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.engines.shared.algebraist.stencil import AlgebraistStencil
from stark.engines.shared.algebraist.allocator import AlgebraistAllocator
from stark.engines.shared.algebraist.frame import AlgebraistFrame
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
    frame: AlgebraistFrame | None = None
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

    def provide_delta(self, request: AlgebraistStencil) -> Callable[..., TranslationType]:
        return self._support.provide_delta(request=request)

    def provide_apply(self, request: AlgebraistStencil) -> Callable[..., StateType]:
        return self._support.provide_apply(request=request)


__all__ = ["AlgebraistRuntimeSpecialist"]
