from __future__ import annotations

from stark.core.auditor import Auditor
from stark.core.contracts import DerivativeIMEX, State, Allocator
from stark.methods.schemes.display.display import display_imex_resolvent_problem
from stark.methods.schemes.execution.step_support import SchemeStepSupport


def initialise_imex_support(
    scheme,
    derivative: DerivativeIMEX,
    allocator: Allocator,
) -> SchemeStepSupport:
    translation_probe = allocator.allocate_translation()
    Auditor.require_imex_scheme_inputs(derivative, allocator, translation_probe)
    scheme.workspace = SchemeStepSupport(allocator, translation_probe)
    return scheme.workspace


def imex_display_resolvent_problem(cls) -> str:
    """Return the standard display text for an IMEX resolvent problem."""

    return display_imex_resolvent_problem(
        cls.tableau,
        cls.descriptor.short_name,
        cls.descriptor.full_name,
    )


def imex_snapshot_state(self, state: State) -> State:
    """Return an IMEX scheme-owned snapshot of *state*."""

    return self.workspace.snapshot_state(state)


__all__ = [
    "imex_display_resolvent_problem",
    "imex_snapshot_state",
    "initialise_imex_support",
]
