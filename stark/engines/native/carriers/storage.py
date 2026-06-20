from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, TypeAlias

from stark.engines.native.carriers.array import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.engines.native.carriers.list import CarrierNativeListValue, CarrierStorageNativeList
from stark.engines.native.carriers.scalar import CarrierNativeScalarValue, CarrierStorageNativeScalar
from stark.engines.native.carriers.tuple import CarrierNativeTupleValue, CarrierStorageNativeTuple

CarrierNativeValue: TypeAlias = (
    CarrierNativeScalarValue
    | CarrierNativeListValue
    | CarrierNativeTupleValue
    | CarrierNativeArrayValue
)


@dataclass(frozen=True)
class CarrierStorageNative:
    """Compatibility facade for code that imports the old native storage class."""

    template: CarrierNativeValue
    concrete: Any = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.template, Number):
            concrete = CarrierStorageNativeScalar(self.template)
        elif isinstance(self.template, list):
            concrete = CarrierStorageNativeList.from_template(self.template)
        elif isinstance(self.template, tuple):
            concrete = CarrierStorageNativeTuple.from_template(self.template)
        elif isinstance(self.template, array):
            concrete = CarrierStorageNativeArray.from_template(self.template)
        else:
            raise TypeError(
                "Native carrier template must be numeric, list, tuple, or floating array.array."
            )
        object.__setattr__(self, "concrete", concrete)

    def is_state(self, value: CarrierNativeValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNativeValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: object) -> bool:
        return self.concrete.matches_template(value)
