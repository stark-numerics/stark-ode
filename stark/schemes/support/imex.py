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


def with_imex_workspace_methods(cls):
    """Install the standard IMEX workspace and resolvent-display methods."""

    @classmethod
    def display_resolvent_problem(inner_cls) -> str:
        return display_imex_resolvent_problem(
            inner_cls.tableau,
            inner_cls.descriptor.short_name,
            inner_cls.descriptor.full_name,
        )

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.workspace.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.workspace.snapshot_state(state)

    cls.display_resolvent_problem = display_resolvent_problem
    cls.set_apply_delta_safety = set_apply_delta_safety
    cls.snapshot_state = snapshot_state
    return cls


__all__ = ["initialise_imex_support", "with_imex_workspace_methods"]
