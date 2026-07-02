from __future__ import annotations

from typing import Generic

from stark.core.auditor import Auditor
from stark.core.contracts import Allocator, DerivativeLike, StateType, TranslationType
from stark.methods.schemes.execution.step_support import SchemeStepSupport


class SchemeRuntimeExplicit(Generic[StateType, TranslationType]):
    """Runtime binding shared by built-in explicit schemes.

    This object owns the non-algorithmic state that explicit schemes share:
    derivative binding, input audit, step-support construction, the shared
    first translation buffer, and state snapshots. It is generic over the
    engine's state and translation objects so a scheme created for a scalar
    native engine, a NumPy frame, or a generated backend keeps the same concrete
    pair all the way through its workspace.

    Concrete schemes still own their actual step algorithm. They may copy
    selected attributes such as `workspace` and `k1` onto themselves so their
    hot paths stay direct and branch-free.
    """

    __slots__ = ("derivative", "first_translation", "workspace")

    def __init__(
        self,
        derivative: DerivativeLike[StateType, TranslationType],
        allocator: Allocator[StateType, TranslationType],
    ) -> None:
        first_translation = allocator.allocate_translation()
        Auditor.require_scheme_inputs(derivative, allocator, first_translation)
        self.derivative = derivative
        self.workspace = SchemeStepSupport(allocator, first_translation)
        self.first_translation = first_translation

    @property
    def k1(self) -> TranslationType:
        return self.first_translation

    def snapshot_state(self, state: StateType) -> StateType:
        return self.workspace.snapshot_state(state)


__all__ = [
    "SchemeRuntimeExplicit",
]
