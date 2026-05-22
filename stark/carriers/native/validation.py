from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stark.carriers.native.array import CarrierValidationNativeArray
from stark.carriers.native.list import CarrierValidationNativeList
from stark.carriers.native.scalar import CarrierValidationNativeScalar
from stark.carriers.native.storage import CarrierNativeValue, CarrierStorageNative
from stark.carriers.native.tuple import CarrierValidationNativeTuple


@dataclass(frozen=True)
class CarrierValidationNative:
    """Compatibility facade for code that imports the old native validation class."""

    storage: CarrierStorageNative

    def __post_init__(self) -> None:
        concrete_storage = self.storage.concrete
        if concrete_storage.__class__.__name__ == "CarrierStorageNativeScalar":
            concrete = CarrierValidationNativeScalar(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeList":
            concrete = CarrierValidationNativeList(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeTuple":
            concrete = CarrierValidationNativeTuple(concrete_storage)
        elif concrete_storage.__class__.__name__ == "CarrierStorageNativeArray":
            concrete = CarrierValidationNativeArray(concrete_storage)
        else:  # pragma: no cover - defensive against future storage classes.
            raise TypeError("Unsupported native storage class.")
        object.__setattr__(self, "concrete", concrete)

    concrete: Any = field(init=False)

    def validate_state(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.validate_state(value)

    def validate_translation(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.concrete.validate_translation(value)

    def coerce_translation(self, value: object) -> CarrierNativeValue:
        return self.concrete.coerce_translation(value)
