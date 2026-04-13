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

    A translation should behave like an element of the tangent space around a
    state. In practice that means:

    - `translation(origin, result)` applies the update to `origin` and writes
      the updated state into `result`
    - `norm()` measures the size of the update
    - `+` and scalar multiplication provide the linear operations STARK uses in
      explicit and implicit methods
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
    """
    A grouped solver-space object built from one or more translations.

    Implicit schemes, resolvers, and inverters work in a product space of
    translations rather than on single translations alone. A one-stage implicit
    method therefore uses a one-item block, while multi-stage methods and
    quasi-Newton histories can use larger blocks.
    """

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
    """
    Fill `out` with the image of a translation under a linear operator.

    Users normally provide operators through a `Linearizer`. The operator does
    not need to expose a dense matrix. It only needs to apply the local linear
    map to a translation, which is enough for matrix-free inverters such as
    GMRES, FGMRES, and BiCGStab.
    """

    def __call__(self, out: Translation, translation: Translation) -> None:
        ...


class InnerProduct(Protocol):
    """
    Return the inner product of two translations.

    Norms alone are not enough for Krylov methods. If a resolver or inverter
    needs orthogonalization or secant projections, the user must also provide
    an inner product compatible with the translation space.
    """

    def __call__(self, left: Translation, right: Translation) -> float:
        ...


class Linearizer(Protocol):
    """
    Fill `out` with the local Jacobian action of the derivative at `state`.

    This is the contract that asks the user to do some problem-specific maths.
    Given a nonlinear derivative

        x' = f(x),

    the linearizer must provide the action of the Jacobian

        J(state) * translation

    as an `Operator`. STARK does not ask for a dense matrix. It asks for a
    callable linear operator that, given an input translation, writes the
    Jacobian image into `out`.

    Built-in implicit schemes then use that operator to construct the actual
    linearized residual operators they need, such as

        I - dt * J(state)

    for backward Euler or the corresponding stage operators for SDIRK methods.

    So the pencil-on-paper task for the user is: derive the Jacobian action of
    the derivative on a translation in the representation used by the problem.
    """

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
