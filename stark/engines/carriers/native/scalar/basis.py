"""Coordinate bases for native scalar carrier values."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass

from stark.engines.carriers.native.scalar.storage import CarrierNativeScalarValue, CarrierStorageNativeScalar


@dataclass(frozen=True)
class CarrierBasisNativeScalar:
    """Canonical one-dimensional basis for native scalars."""

    storage: CarrierStorageNativeScalar

    @property
    def dimension(self) -> int:
        return 1

    def vector(self, index: int, output: CarrierNativeScalarValue) -> CarrierNativeScalarValue:
        del output
        if index != 0:
            raise IndexError("Native scalar basis index out of range.")
        return 1.0

    def coordinate(self, index: int, translation: CarrierNativeScalarValue) -> float:
        if index != 0:
            raise IndexError("Native scalar basis index out of range.")
        return float(translation)

    def coordinates(
        self,
        translation: CarrierNativeScalarValue,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        output[0] = float(translation)
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: CarrierNativeScalarValue,
    ) -> CarrierNativeScalarValue:
        del output
        return float(coordinates[0])


__all__ = ["CarrierBasisNativeScalar"]
