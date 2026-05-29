from __future__ import annotations

from dataclasses import dataclass

from stark.core.auditor import Auditor
from stark.contracts import Derivative, State, Translation, Allocator
from stark.machinery.stage_solve.workspace import SchemeWorkspace


@dataclass(slots=True)
class SchemeSupportExplicit:
    """Reusable setup support for explicit schemes.

    This object owns the non-algorithmic setup that explicit schemes share:
    derivative binding, input audit, workspace construction, the shared first
    translation buffer, snapshots, and apply-delta safety.

    Concrete schemes should still own their actual step algorithm.
    """

    derivative: Derivative
    workspace: SchemeWorkspace
    first_translation: Translation

    @classmethod
    def from_inputs(
        cls,
        derivative: Derivative,
        allocator: Allocator,
    ) -> SchemeSupportExplicit:
        first_translation = allocator.allocate_translation()
        Auditor.require_scheme_inputs(derivative, allocator, first_translation)

        return cls(
            derivative=derivative,
            workspace=SchemeWorkspace(allocator, first_translation),
            first_translation=first_translation,
        )

    @property
    def k1(self) -> Translation:
        return self.first_translation

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)


def initialise_explicit_support(
    scheme,
    derivative: Derivative,
    allocator: Allocator,
) -> SchemeSupportExplicit:
    """Attach the standard explicit scheme support object to a scheme."""

    support = SchemeSupportExplicit.from_inputs(derivative, allocator)
    scheme.explicit = support
    scheme.derivative = support.derivative
    scheme.workspace = support.workspace
    scheme.k1 = support.k1
    return support


def explicit_set_apply_delta_safety(self, enabled: bool) -> None:
    """Set explicit workspace apply-delta safety for this scheme."""

    self.explicit.set_apply_delta_safety(enabled)


def explicit_snapshot_state(self, state: State) -> State:
    """Return an explicit workspace-owned snapshot of *state*."""

    return self.explicit.snapshot_state(state)


__all__ = [
    "SchemeSupportExplicit",
    "explicit_set_apply_delta_safety",
    "explicit_snapshot_state",
    "initialise_explicit_support",
]
