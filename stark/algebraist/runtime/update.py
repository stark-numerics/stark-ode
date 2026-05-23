from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.runtime.support import AlgebraistRuntimeSupport
from stark.algebraist.update import AlgebraistUpdateKernel, AlgebraistUpdateStencil
from stark.algebraist.workbench import AlgebraistWorkbench
from stark.contracts.acceleration import AcceleratorLike
from stark.contracts.translations import State, Translation

try:
    from stark.algebraist.layout import AlgebraistLayout
except Exception:  # pragma: no cover
    AlgebraistLayout = object  # type: ignore[misc, assignment]

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)


@dataclass(slots=True)
class AlgebraistRuntimeUpdateSpecialist(Generic[StateType, TranslationType]):
    """Runtime provider of fixed-coefficient state-update kernels."""

    translation: TranslationType
    workbench: AlgebraistWorkbench[TranslationType]
    layout: AlgebraistLayout | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)
    _support: AlgebraistRuntimeSupport[TranslationType] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._support = AlgebraistRuntimeSupport(
            translation=self.translation,
            workbench=self.workbench,
            layout=self.layout,
            linear_combine=self.linear_combine,
            accelerator=self.accelerator,
        )

    def provide(self, request: AlgebraistUpdateStencil) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        return self._support.provide_update(request)


__all__ = ["AlgebraistRuntimeUpdateSpecialist"]
