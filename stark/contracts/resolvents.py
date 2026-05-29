"""Contracts for nonlinear resolvent workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.blocks import Block
from stark.contracts.intervals import IntervalLike
from stark.contracts.states import State


class Resolvent(Protocol):
    """
    Solve a scheme-provided implicit stage problem in place.

    A resolvent owns the nonlinear machinery behind equations of the form

        delta - rhs - alpha f(state + delta) = 0

    The scheme packages the stage interval, origin state, known right-hand
    side, diagonal shift, and derivative into a small problem object. The
    resolvent reads that object and writes the solved correction block into
    `out`.
    """

    tableau: Any | None

    def bind(self, interval: IntervalLike, state: State) -> None:
        ...

    def __call__(self, problem: Any, out: Block) -> None:
        ...


class ResolventAudit:
    """Record checks for nonlinear resolvent workers."""

    def __call__(self, recorder: AuditRecorder, resolvent: Any) -> None:
        recorder.check(
            callable(getattr(resolvent, "bind", None)),
            "Resolvent provides bind(interval, state).",
            "Add bind(interval, state) before solving shifted implicit equations.",
        )
        recorder.check(
            callable(resolvent),
            "Resolvent provides __call__(problem, out).",
            "Add __call__(problem, out) to solve the scheme-provided implicit problem.",
        )


__all__ = ["Resolvent", "ResolventAudit"]
