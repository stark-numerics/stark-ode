"""Coordinate bases for native list carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass

from stark.engines.carriers.native.list.storage import CarrierNativeListValue, CarrierStorageNativeList


@dataclass(frozen=True)
class CarrierBasisNativeList:
    """Canonical coordinate basis for native lists."""

    storage: CarrierStorageNativeList

    @property
    def dimension(self) -> int:
        return self.storage.length

    def vector(self, index: int, output: CarrierNativeListValue) -> CarrierNativeListValue:
        del output
        return [1.0 if position == index else 0.0 for position in range(self.storage.length)]

    def coordinate(self, index: int, translation: CarrierNativeListValue) -> float:
        return float(translation[index])

    def coordinates(
        self,
        translation: CarrierNativeListValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        for index in range(self.storage.length):
            output[index] = float(translation[index])
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierNativeListValue,
    ) -> CarrierNativeListValue:
        del output
        return [float(coordinates[index]) for index in range(self.storage.length)]


__all__ = ["CarrierBasisNativeList"]
