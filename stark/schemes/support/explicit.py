from __future__ import annotations

from dataclasses import dataclass

from stark.accelerators.binding import BoundDerivative
from stark.auditor import Auditor
from stark.contracts import Derivative, State, Translation, Workbench
from stark.machinery.stage_solve.workspace import SchemeWorkspace


@dataclass(slots=True)
class SchemeSupportExplicit:
    """Reusable setup support for explicit schemes.

    This object owns the non-algorithmic setup that explicit schemes share:
    derivative binding, input audit, workspace construction, the shared first
    translation buffer, snapshots, and apply-delta safety.

    Concrete schemes should still own their actual step algorithm.
    """

    derivative: BoundDerivative
    workspace: SchemeWorkspace
    first_translation: Translation

    @classmethod
    def from_inputs(
        cls,
        derivative: Derivative,
        workbench: Workbench,
    ) -> SchemeSupportExplicit:
        first_translation = workbench.allocate_translation()
        Auditor.require_scheme_inputs(derivative, workbench, first_translation)

        return cls(
            derivative=BoundDerivative(derivative),
            workspace=SchemeWorkspace(workbench, first_translation),
            first_translation=first_translation,
        )

    @property
    def k1(self) -> Translation:
        return self.first_translation

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)


__all__ = ["SchemeSupportExplicit"]