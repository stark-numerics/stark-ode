from __future__ import annotations

from stark.accelerators.binding import DerivativeAccelerated
from stark.block import BlockAllocator
from stark.contracts import Derivative, State, Workbench
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.schemes.support.display import display_implicit_resolvent_problem


def initialise_implicit_support(self, derivative: Derivative, workbench: Workbench) -> None:
    """Initialise shared support for built-in implicit schemes."""

    validate_implicit_resolvent_tableau(self)
    translation_probe = workbench.allocate_translation()
    self.derivative = DerivativeAccelerated(derivative)
    self.workspace = SchemeWorkspace(workbench, translation_probe)
    self.block_allocator = BlockAllocator(workbench)


def validate_implicit_resolvent_tableau(self) -> None:
    """Check any declared resolvent tableau against the scheme tableau."""

    resolvent = getattr(self, "resolvent", None)
    scheme_tableau = getattr(self, "tableau", None)
    resolvent_tableau = getattr(resolvent, "tableau", None)

    if (
        scheme_tableau is not None
        and resolvent_tableau is not None
        and resolvent_tableau != scheme_tableau
    ):
        scheme_name = getattr(getattr(self, "descriptor", None), "short_name", type(self).__name__)
        raise ValueError(
            f"{scheme_name} requires a resolvent configured with its own tableau."
        )


def with_implicit_workspace_methods(cls):
    """Install standard implicit workspace delegation methods."""

    @classmethod
    def display_resolvent_problem(inner_cls) -> str:
        return display_implicit_resolvent_problem(
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


# Compatibility alias while existing imports are rolled over.
with_implicit_stepper_methods = with_implicit_workspace_methods


__all__ = [
    "initialise_implicit_support",
    "validate_implicit_resolvent_tableau",
    "with_implicit_stepper_methods",
    "with_implicit_workspace_methods",
]
