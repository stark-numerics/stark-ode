"""Coordinate bases for CuPy-backed carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from math import prod

from stark.carriers.cupy.storage import CarrierCupyValue, CarrierStorageCupy


@dataclass(frozen=True)
class CarrierBasisCupy:
    """Canonical coordinate basis for a CuPy carrier."""

    storage: CarrierStorageCupy

    @property
    def dimension(self) -> int:
        return prod(self.storage.shape)

    def vector(self, index: int, output: CarrierCupyValue) -> CarrierCupyValue:
        output[...] = 0.0
        output.reshape(-1)[index] = 1.0
        return output

    def coordinate(self, index: int, translation: CarrierCupyValue) -> float:
        return float(translation.reshape(-1)[index].item())

    def coordinates(
        self,
        translation: CarrierCupyValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        flat = translation.reshape(-1)
        for index in range(self.dimension):
            output[index] = float(flat[index].item())
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierCupyValue,
    ) -> CarrierCupyValue:
        output[...] = 0.0
        flat = output.reshape(-1)
        for index, coordinate in enumerate(coordinates):
            flat[index] = coordinate
        return output


__all__ = ["CarrierBasisCupy"]
