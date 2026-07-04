"""Contracts for nonlinear resolvent workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder


class Resolvent(Protocol):
    """
    Solve a scheme-provided implicit stage problem in place.

    A resolvent owns the nonlinear machinery behind equations of the form

        delta - rhs - alpha f(state + delta) = 0

    The scheme packages the stage interval, origin state, known right-hand
    side, diagonal shift, and derivative into a small problem object. The
    resolvent reads that object and returns the solved correction block.

    The broad contract intentionally leaves the concrete request and block
    types as ``Any``. Built-in resolvents carry precise generics internally,
    but schemes accept this public handshake because different resolvent
    families use different request shapes.
    """

    tableau: Any | None

    def __call__(self, problem: Any, delta: Any) -> Any:
        ...


class ResolventAudit:
    """Record checks for nonlinear resolvent workers."""

    def __call__(self, recorder: AuditRecorder, resolvent: Any) -> None:
        recorder.check(
            callable(resolvent),
            "Resolvent provides __call__(problem, delta).",
            "Add __call__(problem, delta) returning the solved correction block.",
        )


__all__ = ["Resolvent", "ResolventAudit"]
