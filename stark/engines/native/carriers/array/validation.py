from __future__ import annotations

from array import array
from dataclasses import dataclass
from numbers import Number
from collections.abc import Iterable

from stark.engines.native.carriers.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray


@dataclass(frozen=True)
class CarrierValidationNativeArray:
    storage: CarrierStorageNativeArray

    def validate_state(self, value: CarrierNativeArrayValue) -> CarrierNativeArrayValue:
        return self.validate_value(value, "state")

    def validate_translation(self, value: CarrierNativeArrayValue) -> CarrierNativeArrayValue:
        return self.validate_value(value, "translation")

    def coerce_translation(self, value: object) -> CarrierNativeArrayValue:
        if isinstance(value, array):
            return self.validate_translation(value)
        if not isinstance(value, Iterable):
            raise TypeError("Native array carrier translation must be array.array or iterable.")
        coerced = array(self.storage.typecode, value)
        return self.validate_translation(coerced)

    def validate_value(self, value: object, role: str) -> CarrierNativeArrayValue:
        if not isinstance(value, array):
            raise TypeError(f"Native array carrier {role} must be an array.array.")
        if value.typecode != self.storage.typecode:
            raise TypeError(
                f"Native array carrier {role} typecode {value.typecode!r} does not "
                f"match template typecode {self.storage.typecode!r}."
            )
        if len(value) != self.storage.length:
            raise ValueError(f"Native array carrier {role} length does not match template.")
        if not all(isinstance(item, Number) for item in value):
            raise TypeError(f"Native array carrier {role} must contain numeric values.")
        return value
