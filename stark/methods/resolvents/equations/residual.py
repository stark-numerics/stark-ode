from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.block import Block
from stark.core.contracts import Translation, OperatorType

class ResolventResidual(Protocol[OperatorType]):
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
        out: OperatorType,
    ) -> None:
        ...


__all__ = ["ResolventResidual"]
