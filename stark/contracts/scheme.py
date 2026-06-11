"""Contracts for one-step integration schemes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from stark.contracts.contract_audit import AuditRecorder
from stark.contracts.interval import IntervalLike
from stark.contracts.state import State

if TYPE_CHECKING:
    from stark.methods.schemes.method.descriptor import SchemeDescriptor


class SchemeLike(Protocol):
    """
    Minimal protocol accepted by STARK for one-step integration schemes.

    A custom scheme does not need to inherit from STARK's internal helper
    classes as long as it satisfies this interface.

    The scheme is responsible for mutating `state` forward by one accepted step
    and returning the step size that was actually taken. Adaptive schemes may
    also update `interval.step` to propose the next step size.
    """

    def __call__(self, interval: IntervalLike, state: State) -> float:
        ...

    def snapshot_state(self, state: State) -> State:
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
            "Scheme provides __call__(interval, state).",
            "Add __call__(interval, state) returning the accepted step size.",
        )
        recorder.check(
            callable(getattr(scheme, "snapshot_state", None)),
            "Scheme provides snapshot_state(state).",
            "Add snapshot_state(state) so snapshot integration can clone the state.",
        )


__all__ = ["Scheme", "SchemeAudit", "SchemeLike"]
