"""Facade coordinate basis for native carriers."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

from stark.engines.native.carriers.array import CarrierStorageNativeArray
from stark.engines.native.carriers.array.basis import CarrierBasisNativeArray
from stark.engines.native.carriers.list import CarrierStorageNativeList
from stark.engines.native.carriers.list.basis import CarrierBasisNativeList
from stark.engines.native.carriers.scalar import CarrierStorageNativeScalar
from stark.engines.native.carriers.scalar.basis import CarrierBasisNativeScalar
from stark.engines.native.carriers.storage import CarrierNativeValue, CarrierStorageNative
from stark.engines.native.carriers.tuple import CarrierStorageNativeTuple
from stark.engines.native.carriers.tuple.basis import CarrierBasisNativeTuple


@dataclass(frozen=True)
class CarrierBasisNative:
    """Facade that selects the native basis matching the storage shape."""

    storage: CarrierStorageNative | CarrierStorageNativeScalar | CarrierStorageNativeList | CarrierStorageNativeTuple | CarrierStorageNativeArray
    concrete: Any = field(init=False)

    def __post_init__(self) -> None:
        storage = self.storage.concrete if isinstance(self.storage, CarrierStorageNative) else self.storage
        if isinstance(storage, CarrierStorageNativeScalar):
            concrete = CarrierBasisNativeScalar(storage)
        elif isinstance(storage, CarrierStorageNativeList):
            concrete = CarrierBasisNativeList(storage)
        elif isinstance(storage, CarrierStorageNativeTuple):
            concrete = CarrierBasisNativeTuple(storage)
        elif isinstance(storage, CarrierStorageNativeArray):
            concrete = CarrierBasisNativeArray(storage)
        else:  # pragma: no cover - defensive branch for external storage implementations.
            raise TypeError("Unsupported native carrier storage for basis construction.")
        object.__setattr__(self, "concrete", concrete)

    @classmethod
    def from_template(cls, template: CarrierNativeValue) -> "CarrierBasisNative":
        if isinstance(template, Number):
            return cls(CarrierStorageNativeScalar(template))
        if isinstance(template, list):
            return cls(CarrierStorageNativeList.from_template(template))
        if isinstance(template, tuple):
            return cls(CarrierStorageNativeTuple.from_template(template))
        if isinstance(template, array):
            return cls(CarrierStorageNativeArray.from_template(template))
        raise TypeError("Native carrier basis template must be numeric, list, tuple, or floating array.array.")

    @property
    def dimension(self) -> int:
        return self.concrete.dimension

    def vector(self, index: int, output: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.vector(index, output)

    def coordinate(self, index: int, translation: CarrierNativeValue) -> float:
        return self.concrete.coordinate(index, translation)

    def coordinates(self, translation: CarrierNativeValue, output):
        return self.concrete.coordinates(translation, output)

    def synthesize(self, coordinates, output: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.synthesize(coordinates, output)


__all__ = ["CarrierBasisNative"]
