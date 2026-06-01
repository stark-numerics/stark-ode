"""Coordinate bases for native tuple carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass

from stark.carriers.native.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple


@dataclass(frozen=True)
class CarrierBasisNativeTuple:
    """Canonical coordinate basis for native tuples."""

    storage: CarrierStorageNativeTuple

    @property
    def dimension(self) -> int:
        return self.storage.length

    def vector(self, index: int, output: CarrierNativeTupleValue) -> CarrierNativeTupleValue:
        del output
        return tuple(1.0 if position == index else 0.0 for position in range(self.storage.length))

    def coordinate(self, index: int, translation: CarrierNativeTupleValue) -> float:
        return float(translation[index])

    def coordinates(
        self,
        translation: CarrierNativeTupleValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        for index in range(self.storage.length):
            output[index] = float(translation[index])
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierNativeTupleValue,
    ) -> CarrierNativeTupleValue:
        del output
        return tuple(float(coordinates[index]) for index in range(self.storage.length))


__all__ = ["CarrierBasisNativeTuple"]
