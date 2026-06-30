from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.block import Block
from stark.core.contracts import Operator, Translation

OperatorTypeContravariant = TypeVar(
    "OperatorTypeContravariant",
    bound=Operator,
    contravariant=True,
)


class ResolventResidual(Protocol[OperatorTypeContravariant]):
    """Residual object used by new-paradigm resolvents."""

    def __call__(
        self,
        delta: Block[Translation],
        out: Block[Translation],
    ) -> None:
        ...

    def differential(
        self,
        delta: Block[Translation],
        out: OperatorTypeContravariant,
    ) -> None:
        ...


__all__ = ["ResolventResidual"]
