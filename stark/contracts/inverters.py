"""Contracts for linear inverter and preconditioner workers."""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.blocks import Block


class InverterLike(Protocol):
    """
    Bind a linear operator and then approximately solve with it.

    Inverters are the linear inner workers used by Newton-like resolvents. They
    do not form explicit inverses. Instead they are configured with an operator
    and then apply an approximate inverse action to a right-hand side block.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, rhs: Block, out: Block) -> None:
        ...


class InverterPreconditionerLike(Protocol):
    """
    Bind a linear operator and apply an approximate inverse-like action.

    STARK treats preconditioners as configured workers with the same broad call
    shape as inverters: they may inspect the operator at bind time, cache any
    scratch they need, and then approximately solve or smooth a block right-
    hand side when called.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, rhs: Block, out: Block) -> None:
        ...


class InverterAudit:
    """Record checks for linear inverters and preconditioners."""

    def __call__(self, recorder: AuditRecorder, inverter: Any) -> None:
        recorder.check(
            callable(getattr(inverter, "bind", None)),
            "Inverter provides bind(operator).",
            "Add bind(operator) so the inverter can prepare a linear solve.",
        )
        recorder.check(
            callable(inverter),
            "Inverter provides __call__(rhs, out).",
            "Add __call__(rhs, out) to apply the approximate inverse action.",
        )

    @staticmethod
    def preconditioner(recorder: AuditRecorder, preconditioner: Any) -> None:
        recorder.check(
            callable(getattr(preconditioner, "bind", None)),
            "InverterPreconditioner provides bind(operator).",
            "Add bind(operator) so the preconditioner can inspect the operator.",
        )
        recorder.check(
            callable(preconditioner),
            "InverterPreconditioner provides __call__(rhs, out).",
            "Add __call__(rhs, out) to apply the preconditioning action.",
        )


__all__ = ["InverterAudit", "InverterLike", "InverterPreconditionerLike"]
