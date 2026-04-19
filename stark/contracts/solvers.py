from __future__ import annotations

"""
Protocols for implicit-solve and linear-solve workers.

This file groups the contracts a user typically reaches for once they move
beyond explicit schemes and start supplying custom residuals, resolvents,
inverters, or preconditioners.
"""

from typing import Any, Protocol

from stark.contracts.audit_support import AuditRecorder

from stark.contracts.intervals import IntervalLike
from stark.contracts.translations import Block, State


class Residual(Protocol):
    """
    Fill `out` with the nonlinear residual evaluated at `block`.

    A nonlinear implicit solve searches for a block whose residual is
    approximately zero. For one-stage implicit schemes this is often a one-item
    block holding a single translation. For multi-stage methods it can hold
    several coupled stage translations.
    """

    def __call__(self, out: Block, block: Block) -> None:
        ...


class LinearResidual(Residual, Protocol):
    """
    Residual that can also linearize itself at a trial block.

    Newton-style resolvents require more than residual evaluation: they also
    need the local linearization of that residual around the current trial
    block. The residual owns that construction because it knows the scheme
    context, step size, and any algebra needed to wrap the user's `Linearizer`
    into the correct residual operator.
    """

    def linearize(self, out: Any, block: Block) -> None:
        ...


class Resolvent(Protocol):
    """
    Bind a stage context and solve a shifted implicit equation in place.

    A resolvent owns the nonlinear machinery behind equations of the form

        delta - rhs - alpha f(state + delta) = 0

    for a bound `interval` and `state`. Schemes bind the stage context and then
    ask the resolvent to write the solved correction block into `out`.
    """

    tableau: Any | None

    def bind(self, interval: IntervalLike, state: State) -> None:
        ...

    def __call__(self, out: Block, alpha: float, rhs: Block | None = None) -> None:
        ...


class InverterLike(Protocol):
    """
    Bind a linear operator and then approximately solve with it.

    Inverters are the linear inner workers used by Newton-like resolvents. They
    do not form explicit inverses. Instead they are configured with an operator
    and then apply an approximate inverse action to a right-hand side block.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, out: Block, rhs: Block) -> None:
        ...


class PreconditionerLike(Protocol):
    """
    Bind a linear operator and apply an approximate inverse-like action.

    STARK treats preconditioners as configured workers with the same broad call
    shape as inverters: they may inspect the operator at bind time, cache any
    scratch they need, and then approximately solve or smooth a block right-
    hand side when called.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, out: Block, rhs: Block) -> None:
        ...


class SolverAudit:
    @staticmethod
    def residual(recorder: AuditRecorder, residual: Any, *, linear: bool = False) -> None:
        recorder.check(
            callable(residual),
            "Residual provides __call__(out, block).",
            "Add __call__(out, block) to evaluate the nonlinear residual.",
        )
        if linear:
            recorder.check(
                callable(getattr(residual, "linearize", None)),
                "Residual provides linearize(out, block).",
                "Add linearize(out, block) for Newton-style resolvents.",
            )

    @staticmethod
    def resolvent(recorder: AuditRecorder, resolvent: Any) -> None:
        recorder.check(
            callable(getattr(resolvent, "bind", None)),
            "Resolvent provides bind(interval, state).",
            "Add bind(interval, state) before solving shifted implicit equations.",
        )
        recorder.check(
            callable(resolvent),
            "Resolvent provides __call__(out, alpha, rhs=None).",
            "Add __call__(out, alpha, rhs=None) to solve the bound implicit problem.",
        )

    @staticmethod
    def inverter(recorder: AuditRecorder, inverter: Any) -> None:
        recorder.check(
            callable(getattr(inverter, "bind", None)),
            "Inverter provides bind(operator).",
            "Add bind(operator) so the inverter can prepare a linear solve.",
        )
        recorder.check(
            callable(inverter),
            "Inverter provides __call__(out, rhs).",
            "Add __call__(out, rhs) to apply the approximate inverse action.",
        )

    @staticmethod
    def preconditioner(recorder: AuditRecorder, preconditioner: Any) -> None:
        recorder.check(
            callable(getattr(preconditioner, "bind", None)),
            "Preconditioner provides bind(operator).",
            "Add bind(operator) so the preconditioner can inspect the operator.",
        )
        recorder.check(
            callable(preconditioner),
            "Preconditioner provides __call__(out, rhs).",
            "Add __call__(out, rhs) to apply the preconditioning action.",
        )


__all__ = [
    "InverterLike",
    "SolverAudit",
    "LinearResidual",
    "PreconditionerLike",
    "Residual",
    "Resolvent",
]









