from dataclasses import dataclass
from numbers import Number

from stark.carriers.native.storage import CarrierNativeValue, CarrierStorageNative


@dataclass(frozen=True)
class CarrierValidationNative:
    storage: CarrierStorageNative

    def validate_state(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.validate_value(value, "state")

    def validate_translation(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.validate_value(value, "translation")

    def coerce_translation(self, value: CarrierNativeValue) -> CarrierNativeValue:
        return self.validate_translation(value)

    def validate_value(self, value: CarrierNativeValue, role: str) -> CarrierNativeValue:
        if not self.storage.matches_template(value):
            raise TypeError(f"Native carrier {role} does not match template storage.")

        if isinstance(value, Number):
            return value

        if isinstance(value, list | tuple):
            if all(isinstance(item, Number) for item in value):
                return value

        raise TypeError(f"Native carrier {role} must contain numeric values.")