from __future__ import annotations

from stark.core.auditor import Auditor
from stark.core.contracts import DerivativeLike, State, Translation, Allocator
from stark.methods.schemes.execution.step_support import SchemeStepSupport


class SchemeRuntimeExplicit:
    """Runtime machinery shared by built-in explicit schemes.

    This object owns the non-algorithmic state that explicit schemes share:
    derivative binding, input audit, step-support construction, the shared
    first translation buffer, and state snapshots.

    Concrete schemes still own their actual step algorithm. They may copy
    selected attributes such as `workspace` and `k1` onto themselves so their
    hot paths stay direct and branch-free.
    """

    __slots__ = ("derivative", "first_translation", "workspace")

    def __init__(
        self,
        derivative: DerivativeLike,
        allocator: Allocator,
    ) -> None:
        first_translation = allocator.allocate_translation()
        Auditor.require_scheme_inputs(derivative, allocator, first_translation)
        self.derivative = derivative
        self.workspace = SchemeStepSupport(allocator, first_translation)
        self.first_translation = first_translation

    @property
    def k1(self) -> Translation:
        return self.first_translation

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)


__all__ = [
    "SchemeRuntimeExplicit",
]
