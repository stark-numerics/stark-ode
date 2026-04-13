from __future__ import annotations

from typing import Any, Protocol

from stark.contract_protocols.linear_algebra import Block


class Residual(Protocol):
    """
    Fill `out` with the nonlinear residual evaluated at `block`.

    A resolver searches for a block whose residual is approximately zero. For
    one-stage implicit schemes this is often a one-item block holding a single
    translation. For multi-stage methods it can hold several coupled stage
    translations.
    """

    def __call__(self, out: Block, block: Block) -> None:
        ...


class LinearResidual(Residual, Protocol):
    """
    Residual that can also linearize itself at a trial block.

    Newton-style resolvers require more than residual evaluation: they also
    need the local linearization of that residual around the current trial
    block. The residual owns that construction because it knows the scheme
    context, step size, and any algebra needed to wrap the user's `Linearizer`
    into the correct residual operator.
    """

    def linearize(self, out: Any, block: Block) -> None:
        ...


class ResolverLike(Protocol):
    """
    Mutate `block` until it approximately resolves the residual equation.

    A resolver works at the nonlinear level. Picard, Anderson, Broyden, and
    Newton all satisfy this contract, even though some require more structure
    than others from the residual.
    """

    def __call__(self, block: Block, residual: Residual) -> None:
        ...


class InverterLike(Protocol):
    """
    Bind a linear operator and then approximately solve with it.

    Inverters are the linear inner workers used by Newton-like resolvers. They
    do not form explicit inverses. Instead they are configured with an operator
    and then apply an approximate inverse action to a right-hand side block.
    """

    def bind(self, operator: Any) -> None:
        ...

    def __call__(self, out: Block, rhs: Block) -> None:
        ...


__all__ = ["InverterLike", "LinearResidual", "Residual", "ResolverLike"]
