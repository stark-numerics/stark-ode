"""Coordinate bases derived from engine frame/carrier pairs.

This module is intentionally engine support rather than ordinary user-facing
problem API. An engine translation basis is useful for inspection, dense
operator materialisation, and diagnostic tools that need coordinates for
a backend-owned translation object. Most application code should not need to
touch it directly.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

from stark.core.contracts.frame import FrameLike


@dataclass(frozen=True, slots=True)
class EngineTranslationBasis:
    """Coordinate basis for a complete engine-owned translation object.

    The basis is assembled from the canonical carrier basis for each frame
    field. It exists primarily for inspection and dense materialisation paths:
    dense inverters can use it to turn backend translations into compact
    coordinate vectors without asking users to write coordinate-basis classes
    by hand.
    """

    algebraist_frame: FrameLike
    carriers: tuple[Any, ...]
    offsets: tuple[int, ...] = field(init=False)
    dimension: int = field(init=False)

    def __post_init__(self) -> None:
        offsets = tuple(
            accumulate((carrier.basis.dimension for carrier in self.carriers), initial=0)
        )
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "dimension", offsets[-1])

    def vector(self, index: int, output: Any) -> Any:
        """Write or return the selected translation basis vector."""

        field_index, local_index = self.local_index(index)
        for current, (field, carrier) in enumerate(
            zip(self.algebraist_frame.fields, self.carriers, strict=True)
        ):
            value = field.translation_path(output)
            if current == field_index:
                value = carrier.basis.vector(local_index, value)
            else:
                value = carrier.basis.synthesize([0.0] * carrier.basis.dimension, value)
            field.translation_path.assign(output, value)
        return output

    def coordinate(self, index: int, translation: Any) -> float:
        """Apply the selected coordinate form to a translation."""

        field_index, local_index = self.local_index(index)
        field = self.algebraist_frame.fields[field_index]
        carrier = self.carriers[field_index]
        return carrier.basis.coordinate(
            local_index,
            field.translation_path(translation),
        )

    def coordinates(
        self,
        translation: Any,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        """Analyse a translation into flat coordinates in this basis."""

        for index, (field, carrier) in enumerate(
            zip(self.algebraist_frame.fields, self.carriers, strict=True)
        ):
            start = self.offsets[index]
            local = [0.0] * carrier.basis.dimension
            carrier.basis.coordinates(field.translation_path(translation), local)
            for local_index, coordinate in enumerate(local):
                output[start + local_index] = coordinate
        return output

    def synthesize(self, coordinates: Sequence[float], output: Any) -> Any:
        """Reconstruct a translation from flat coordinates in this basis."""

        for index, (field, carrier) in enumerate(
            zip(self.algebraist_frame.fields, self.carriers, strict=True)
        ):
            start = self.offsets[index]
            stop = self.offsets[index + 1]
            value = carrier.basis.synthesize(
                coordinates[start:stop],
                field.translation_path(output),
            )
            field.translation_path.assign(output, value)
        return output

    def local_index(self, index: int) -> tuple[int, int]:
        """Return the field and local carrier coordinate for a flat index."""

        if index < 0 or index >= self.dimension:
            raise IndexError("Engine translation basis index out of range.")

        for field_index in range(len(self.carriers)):
            start = self.offsets[field_index]
            stop = self.offsets[field_index + 1]
            if index < stop:
                return field_index, index - start

        raise IndexError("Engine translation basis index out of range.")


__all__ = ["EngineTranslationBasis"]
