"""Contracts for one-step integration schemes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.intervals import IntervalLike
from stark.contracts.states import State

if TYPE_CHECKING:
    from stark.schemes.support.descriptor import SchemeDescriptor
    from stark.schemes.support.executor import SchemeExecutor


class SchemeLike(Protocol):
    """
    Minimal protocol accepted by STARK for one-step integration schemes.

    A custom scheme does not need to inherit from STARK's internal helper
    classes as long as it satisfies this interface.

    The scheme is responsible for mutating `state` forward by one accepted step
    and returning the step size that was actually taken. Adaptive schemes may
    also update `interval.step` to propose the next step size.
    """

    def __call__(self, interval: IntervalLike, state: State, executor: SchemeExecutor) -> float:
        ...

    def snapshot_state(self, state: State) -> State:
        ...

    def set_apply_delta_safety(self, enabled: bool) -> None:
        ...


class Scheme(SchemeLike, Protocol):
    """
    Richer scheme protocol for STARK's built-in, tableau-backed schemes.

    This adds the descriptor and tableau metadata used for readable reporting,
    table display, and package-level exports.
    """

    descriptor: SchemeDescriptor
    tableau: Any

    @classmethod
    def display_tableau(cls) -> str:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __format__(self, format_spec: str) -> str:
        ...


class SchemeAudit:
    """Audit a one-step scheme object."""

    @staticmethod
    def __call__(recorder: AuditRecorder, scheme: Any) -> None:
        recorder.check(
            callable(scheme),
            "Scheme provides __call__(interval, state, executor).",
            "Add __call__(interval, state, executor) returning the accepted step size.",
        )
        recorder.check(
            callable(getattr(scheme, "snapshot_state", None)),
            "Scheme provides snapshot_state(state).",
            "Add snapshot_state(state) so snapshot integration can clone the state.",
        )
        recorder.check(
            callable(getattr(scheme, "set_apply_delta_safety", None)),
            "Scheme provides set_apply_delta_safety(enabled).",
            "Add set_apply_delta_safety(enabled) to control alias-safe state updates.",
        )


__all__ = ["Scheme", "SchemeAudit", "SchemeLike"]
