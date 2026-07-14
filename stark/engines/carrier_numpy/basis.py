"""Coordinate bases for NumPy-backed carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from math import prod

from stark.engines.carrier_numpy.storage import CarrierNumpyValue, CarrierStorageNumpy


@dataclass(frozen=True)
class CarrierBasisNumpy:
    """Canonical coordinate basis for a NumPy carrier."""

    storage: CarrierStorageNumpy

    @property
    def dimension(self) -> int:
        return prod(self.storage.shape)

    def vector(self, index: int, output: CarrierNumpyValue) -> CarrierNumpyValue:
        output[...] = 0.0
        output.flat[index] = 1.0
        return output

    def coordinate(self, index: int, translation: CarrierNumpyValue) -> float:
        return float(translation.flat[index])

    def coordinates(
        self,
        translation: CarrierNumpyValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        for index in range(self.dimension):
            output[index] = float(translation.flat[index])
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierNumpyValue,
    ) -> CarrierNumpyValue:
        output[...] = 0.0
        for index, coordinate in enumerate(coordinates):
            output.flat[index] = coordinate
        return output


__all__ = ["CarrierBasisNumpy"]
