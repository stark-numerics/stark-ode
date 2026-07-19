"""Contracts for coordinate bases on translation spaces."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Protocol, TypeVar

TranslationBasisValueType = TypeVar("TranslationBasisValueType")
"""Type variable for values described by a coordinate basis.

Coordinate bases are used both for full STARK translation objects and for
backend carrier values that live inside a structured translation. A carrier
value may be a list, tuple, NumPy array, CuPy array, JAX array, or scalar, so
this type variable is intentionally not bound to the public `Translation`
protocol.
"""


class TranslationBasisLike(Protocol[TranslationBasisValueType]):
    """
    Coordinate basis for a translation space.

    A translation basis supplies basis vectors and matching coordinate forms.
    Dense operator materialisation can use these operations to build matrix
    entries by applying an operator to each basis vector and reading coordinates
    from the resulting translation.

    The coordinate operation should be dual to the vector operation:

        coordinate(i, vector(j)) == 1 if i == j else 0

    ``coordinates`` analyses a translation into its coordinate vector, while
    ``synthesize`` reconstructs a translation from coordinates in this basis.

    ``vector`` and ``synthesize`` return the written translation so immutable
    return-style carriers, such as JAX, can satisfy the same contract as
    in-place carriers.
    """

    @property
    def dimension(self) -> int:
        """Number of scalar coordinates in this basis."""
        ...

    def vector(self, index: int, output: TranslationBasisValueType, /) -> TranslationBasisValueType:
        """Write or return the selected basis vector."""
        ...

    def coordinate(self, index: int, translation: TranslationBasisValueType, /) -> float:
        """Apply the selected coordinate form to a translation."""
        ...

    def coordinates(
        self,
        translation: TranslationBasisValueType,
        output: MutableSequence[float],
        /,
    ) -> MutableSequence[float]:
        """Analyse a translation into coordinates in this basis."""
        ...

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: TranslationBasisValueType,
        /,
    ) -> TranslationBasisValueType:
        """Reconstruct a translation from coordinates in this basis."""
        ...


__all__ = ["TranslationBasisValueType", "TranslationBasisLike"]
