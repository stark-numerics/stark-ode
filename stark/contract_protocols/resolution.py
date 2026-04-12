from __future__ import annotations

from typing import Any, Protocol

from stark.contract_protocols.linear_algebra import Block


class Residual(Protocol):
    """Fill `out` with the nonlinear residual evaluated at `block`."""

    def __call__(self, out: Block, block: Block) -> None:
        ...


class LinearResidual(Residual, Protocol):
    """Residual that can also linearize itself at a trial block."""

    def linearize(self, out: Any, block: Block) -> None:
        ...


class ResolverLike(Protocol):
    """Mutate `block` until it approximately resolves the residual equation."""

    def __call__(self, block: Block, residual: Residual) -> None:
        ...


class InverterLike(Protocol):
    """Bind a linear operator and then approximately solve with it."""

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, out: Block, rhs: Block) -> None:
        ...


__all__ = ["InverterLike", "LinearResidual", "Residual", "ResolverLike"]
