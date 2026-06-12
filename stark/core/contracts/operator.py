"""Contracts for matrix-free linear operators."""

from __future__ import annotations

from typing import Protocol, TypeVar

from stark.core.contracts.translation import Translation


class Operator(Protocol):
    """
    Fill `out` with the image of a translation under a linear operator.

    Users normally provide operators through a `Linearizer`. The operator does
    not need to expose a dense matrix. It only needs to apply the local linear
    map to a translation, which is enough for matrix-free inverters such as
    GMRES, FGMRES, and BiCGStab.
    """

    def __call__(self, translation: Translation, out: Translation) -> None:
        ...


OperatorType = TypeVar("OperatorType", bound=Operator)


__all__ = ["Operator", "OperatorType"]
