from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt
from typing import Protocol, TypeAlias, TypeVar, Self

State = TypeVar("State")


class Translation(Protocol):
    """
    A state update object carrying the linear structure of the problem.

    STARK separates nonlinear mutable state from linear translation objects.
    Schemes build weighted combinations of translations, and a translation can
    then be applied to a state to produce an updated state.
    """

    def __call__(self, origin: State, result: State) -> None:
        ...

    def norm(self) -> float:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __rmul__(self, scalar: float) -> Self:
        ...


Derivative: TypeAlias = Callable[[State, Translation], None]


@dataclass(slots=True)
class Block:
    """A grouped solver-space object built from one or more translations."""

    items: list[Translation]

    def __repr__(self) -> str:
        return f"Block(size={len(self.items)!r})"

    def __str__(self) -> str:
        return f"block[{len(self.items)}]"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Translation:
        return self.items[index]

    def norm(self) -> float:
        if not self.items:
            return 0.0
        return sqrt(sum(item.norm() ** 2 for item in self.items))


class Operator(Protocol):
    """Fill `out` with the image of a translation under a linear operator."""

    def __call__(self, out: Translation, translation: Translation) -> None:
        ...


class InnerProduct(Protocol):
    """Return the inner product of two translations."""

    def __call__(self, left: Translation, right: Translation) -> float:
        ...


class Linearizer(Protocol):
    """Fill `out` with a local linear operator evaluated at `state`."""

    def __call__(self, out: Operator, state: State) -> None:
        ...


__all__ = [
    "Block",
    "Derivative",
    "InnerProduct",
    "Linearizer",
    "Operator",
    "State",
    "Translation",
]
