"""Coordinate bases for native array.array carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass

from stark.engines.native.carriers.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray


@dataclass(frozen=True)
class CarrierBasisNativeArray:
    """Canonical coordinate basis for native array.array values."""

    storage: CarrierStorageNativeArray

    @property
    def dimension(self) -> int:
        return self.storage.length

    def vector(self, index: int, output: CarrierNativeArrayValue) -> CarrierNativeArrayValue:
        for position in range(self.storage.length):
            output[position] = 1.0 if position == index else 0.0
        return output

    def coordinate(self, index: int, translation: CarrierNativeArrayValue) -> float:
        return float(translation[index])

    def coordinates(
        self,
        translation: CarrierNativeArrayValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        for index in range(self.storage.length):
            output[index] = float(translation[index])
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierNativeArrayValue,
    ) -> CarrierNativeArrayValue:
        for index in range(self.storage.length):
            output[index] = float(coordinates[index])
        return output


__all__ = ["CarrierBasisNativeArray"]
