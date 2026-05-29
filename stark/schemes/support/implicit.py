from __future__ import annotations

from stark.block import BlockAllocator
from stark.contracts import Derivative, State, Allocator
from stark.machinery.stage_solve.workspace import SchemeWorkspace
from stark.schemes.support.display import display_implicit_resolvent_problem


def initialise_implicit_support(self, derivative: Derivative, allocator: Allocator) -> None:
    """Initialise shared support for built-in implicit schemes."""

    validate_implicit_resolvent_tableau(self)
    translation_probe = allocator.allocate_translation()
    self.derivative = derivative
    self.workspace = SchemeWorkspace(allocator, translation_probe)
    self.block_allocator = BlockAllocator(allocator)


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


def implicit_display_resolvent_problem(cls) -> str:
    """Return the standard display text for an implicit resolvent problem."""

    return display_implicit_resolvent_problem(
        cls.tableau,
        cls.descriptor.short_name,
        cls.descriptor.full_name,
    )


def implicit_set_apply_delta_safety(self, enabled: bool) -> None:
    """Set implicit workspace apply-delta safety for this scheme."""

    self.workspace.set_apply_delta_safety(enabled)


def implicit_snapshot_state(self, state: State) -> State:
    """Return an implicit workspace-owned snapshot of *state*."""

    return self.workspace.snapshot_state(state)


__all__ = [
    "initialise_implicit_support",
    "validate_implicit_resolvent_tableau",
    "implicit_display_resolvent_problem",
    "implicit_set_apply_delta_safety",
    "implicit_snapshot_state",
]
