from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.engines.algebraist.stencil import AlgebraistStencil
from stark.engines.algebraist.allocator import AlgebraistAllocator
from stark.contracts.accelerator import Accelerator
from stark.contracts.state import State
from stark.contracts.translation import Translation

try:
    from stark.engines.algebraist.frame import AlgebraistFrame
except Exception:  # pragma: no cover
    AlgebraistFrame = object  # type: ignore[misc, assignment]

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(slots=True)
class AlgebraistRuntimeSpecialist(Generic[StateType, TranslationType]):
    """Runtime provider of fixed-coefficient scheme kernels.

    ``stencil.apply`` selects whether the provided kernel writes a translation
    delta or applies that delta to an origin state.
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

    def provide(self, request: AlgebraistStencil) -> Callable[..., object]:
        return self._support.provide_specialist(request)


__all__ = ["AlgebraistRuntimeSpecialist"]
