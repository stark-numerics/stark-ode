from __future__ import annotations

from stark.core.auditor import Auditor
from stark.contracts import DerivativeIMEX, State, Allocator
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.schemes.support.display import display_imex_resolvent_problem


def initialise_imex_support(
    scheme,
    derivative: DerivativeIMEX,
    allocator: Allocator,
) -> SchemeWorkspace:
    translation_probe = allocator.allocate_translation()
    Auditor.require_imex_scheme_inputs(derivative, allocator, translation_probe)
    scheme.workspace = SchemeWorkspace(allocator, translation_probe)
    return scheme.workspace


def imex_display_resolvent_problem(cls) -> str:
    """Return the standard display text for an IMEX resolvent problem."""

    return display_imex_resolvent_problem(
        cls.tableau,
        cls.descriptor.short_name,
        cls.descriptor.full_name,
    )


def imex_set_apply_delta_safety(self, enabled: bool) -> None:
    """Set IMEX workspace apply-delta safety for this scheme."""

    self.workspace.set_apply_delta_safety(enabled)


def imex_snapshot_state(self, state: State) -> State:
    """Return an IMEX workspace-owned snapshot of *state*."""

    return self.workspace.snapshot_state(state)


__all__ = [
    "imex_display_resolvent_problem",
    "imex_set_apply_delta_safety",
    "imex_snapshot_state",
    "initialise_imex_support",
]
