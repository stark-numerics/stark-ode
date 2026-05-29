"""Contracts and type variables for translation objects.

Translation objects carry the linear update structure used by schemes,
resolvents, and inverters. They are separate from user state objects so STARK
can build weighted update combinations without needing to understand the
concrete state layout.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, TypeVar


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

    def __call__(self, origin: Any, result: Any) -> None:
        ...

    def norm(self) -> float:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __rmul__(self, scalar: float) -> Self:
        ...


# Use invariant types when a protocol both accepts and returns the type.
# Example: validate_translation(value: TranslationType) -> TranslationType.
TranslationType = TypeVar("TranslationType", bound=Translation)

# Use covariant types when a protocol only returns the type.
# Example: zero_translation() -> TranslationTypeCovariant.
TranslationTypeCovariant = TypeVar(
    "TranslationTypeCovariant",
    bound=Translation,
    covariant=True,
)

# Use contravariant types when a protocol only accepts the type.
# Example: norm(value: TranslationTypeContravariant) -> float.
TranslationTypeContravariant = TypeVar(
    "TranslationTypeContravariant",
    bound=Translation,
    contravariant=True,
)


__all__ = [
    "Translation",
    "TranslationType",
    "TranslationTypeCovariant",
    "TranslationTypeContravariant",
]
