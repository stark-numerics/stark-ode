from __future__ import annotations

from stark.contracts import State
from stark.schemes.support.display import display_implicit_resolvent_problem


def with_implicit_stepper_methods(cls):
    """Install the standard implicit stepper delegation methods."""

    @classmethod
    def display_resolvent_problem(inner_cls) -> str:
        return display_implicit_resolvent_problem(
            inner_cls.tableau,
            inner_cls.descriptor.short_name,
            inner_cls.descriptor.full_name,
        )

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.stepper.set_apply_delta_safety(enabled)

    def snapshot_state(self, state: State) -> State:
        return self.stepper.snapshot_state(state)

    cls.display_resolvent_problem = display_resolvent_problem
    cls.set_apply_delta_safety = set_apply_delta_safety
    cls.snapshot_state = snapshot_state
    return cls


__all__ = ["with_implicit_stepper_methods"]
