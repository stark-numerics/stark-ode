"""Contracts for coordinate bases on translation spaces."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import Protocol

from stark.contracts.translation import TranslationType


class TranslationBasis(Protocol[TranslationType]):
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

    dimension: int

    def vector(self, index: int, output: TranslationType) -> TranslationType:
        """Write or return the selected basis vector."""
        ...

    def coordinate(self, index: int, translation: TranslationType) -> float:
        """Apply the selected coordinate form to a translation."""
        ...

    def coordinates(
        self,
        translation: TranslationType,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        """Analyse a translation into coordinates in this basis."""
        ...

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: TranslationType,
    ) -> TranslationType:
        """Reconstruct a translation from coordinates in this basis."""
        ...


__all__ = ["TranslationBasis"]
