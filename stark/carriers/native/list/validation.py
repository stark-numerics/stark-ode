from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from stark.carriers.native.list.storage import CarrierNativeListValue, CarrierStorageNativeList


@dataclass(frozen=True)
class CarrierValidationNativeList:
    storage: CarrierStorageNativeList

    def validate_state(self, value: CarrierNativeListValue) -> CarrierNativeListValue:
        return self.validate_value(value, "state")

    def validate_translation(self, value: CarrierNativeListValue) -> CarrierNativeListValue:
        return self.validate_value(value, "translation")

    def coerce_translation(self, value: object) -> CarrierNativeListValue:
        return self.validate_value(value, "translation")

    def validate_value(self, value: object, role: str) -> CarrierNativeListValue:
        if not isinstance(value, list):
            raise TypeError(f"Native list carrier {role} must be a list.")
        if len(value) != self.storage.length:
            raise ValueError(f"Native list carrier {role} length does not match template.")
        if not all(isinstance(item, Number) for item in value):
            raise TypeError(f"Native list carrier {role} must contain numeric values.")
        return value
