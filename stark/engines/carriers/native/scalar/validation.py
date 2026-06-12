from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

from stark.engines.carriers.native.scalar.storage import (
    CarrierNativeScalarValue,
    CarrierStorageNativeScalar,
)


@dataclass(frozen=True)
class CarrierValidationNativeScalar:
    storage: CarrierStorageNativeScalar

    def validate_state(self, value: CarrierNativeScalarValue) -> CarrierNativeScalarValue:
        return self.validate_value(value, "state")

    def validate_translation(self, value: CarrierNativeScalarValue) -> CarrierNativeScalarValue:
        return self.validate_value(value, "translation")

    def coerce_translation(self, value: object) -> CarrierNativeScalarValue:
        return self.validate_value(value, "translation")

    def validate_value(self, value: object, role: str) -> CarrierNativeScalarValue:
        if not isinstance(value, Number):
            raise TypeError(f"Native scalar carrier {role} must be numeric.")
        return value
