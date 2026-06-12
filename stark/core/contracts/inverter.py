"""Contracts for linear inverter and preconditioner workers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from stark.core.contracts.contract_audit import AuditRecorder
from stark.core.contracts.block import BlockLike, BlockOperatorLike
from stark.core.contracts.translation import Translation, TranslationType


class InverterOutputMode(Enum):
    """
    Contract for how an inverter treats the supplied output block.

    overwrite:
        The inverter computes a fresh solution and replaces output contents.

    improve:
        The inverter treats output as the current guess and improves it in place.
    """

    overwrite = "overwrite"
    improve = "improve"


class InverterRequest(Protocol[TranslationType]):
    """
    Structural contract for a linear solve request consumed by an inverter.

    An inverter request represents the block-valued equation

        operator(solution) = residual

    Attributes:
        operator:
            Linear block action to invert or approximately invert. Calling
            `operator(output, image)` should write the image of `output`
            into `image`.

        residual:
            Right-hand side block in the linear equation.
    """

    operator: BlockOperatorLike[TranslationType]
    residual: BlockLike[TranslationType]


class Inverter(Protocol[TranslationType]):
    """
    Linear problem solver used by resolvents.

    An inverter receives a linear problem request and improves `output` in
    place so that

        request.operator(output) ≈ request.residual
    """
    output_mode: InverterOutputMode

    def __call__(
        self,
        request: InverterRequest[TranslationType],
        output: BlockLike[TranslationType],
    ) -> None:
        ...



class LegacyInverterLike(Protocol):
    """
    Bind a linear operator and then approximately solve with it.

    Inverters are the linear inner workers used by Newton-like resolvents. They
    do not form explicit inverses. Instead they are configured with an operator
    and then apply an approximate inverse action to a right-hand side block.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, rhs: BlockLike, out: BlockLike) -> None:
        ...


class LegacyInverterPreconditionerLike(Protocol):
    """
    Bind a linear operator and apply an approximate inverse-like action.

    STARK treats preconditioners as configured workers with the same broad call
    shape as inverters: they may inspect the operator at bind time, cache any
    scratch they need, and then approximately solve or smooth a block right-
    hand side when called.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, rhs: BlockLike, out: BlockLike) -> None:
        ...


class LegacyInverterAudit:
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


__all__ = [
    "InverterOutputMode",
    "InverterRequest",
    "Inverter",
    "LegacyInverterAudit",
    "LegacyInverterLike",
    "LegacyInverterPreconditionerLike",
]
