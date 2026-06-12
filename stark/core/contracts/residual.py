"""Contracts for residual workers used by nonlinear solves."""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.block import BlockLike


class Residual(Protocol):
    """
    Fill `out` with the nonlinear residual evaluated at `block`.

    A nonlinear implicit solve searches for a block whose residual is
    approximately zero. For one-stage implicit schemes this is often a one-item
    block holding a single translation. For multi-stage methods it can hold
    several coupled stage translations.
    """

    def __call__(self, block: BlockLike, out: BlockLike) -> None:
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

    def linearize(self, block: BlockLike, out: Any) -> None:
        ...


class ResidualAudit:
    """Record checks for nonlinear residual workers."""

    def __call__(self, recorder: AuditRecorder, residual: Any, *, linear: bool = False) -> None:
        recorder.check(
            callable(residual),
            "Residual provides __call__(block, out).",
            "Add __call__(block, out) to evaluate the nonlinear residual.",
        )
        if linear:
            recorder.check(
                callable(getattr(residual, "linearize", None)),
                "Residual provides linearize(block, out).",
                "Add linearize(block, out) for Newton-style resolvents.",
            )


__all__ = ["LinearResidual", "Residual", "ResidualAudit"]
