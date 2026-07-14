from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import cast

from stark.engines.carrier_native.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple


@dataclass(frozen=True)
class CarrierValidationNativeTuple:
    storage: CarrierStorageNativeTuple

    def validate_state(self, value: CarrierNativeTupleValue) -> CarrierNativeTupleValue:
        return self.validate_value(value, "state")

    def validate_translation(self, value: CarrierNativeTupleValue) -> CarrierNativeTupleValue:
        return self.validate_value(value, "translation")

    def coerce_translation(self, value: object) -> CarrierNativeTupleValue:
        return self.validate_value(value, "translation")

    def validate_value(self, value: object, role: str) -> CarrierNativeTupleValue:
        if not isinstance(value, tuple):
            raise TypeError(f"Native tuple carrier {role} must be a tuple.")
        if len(value) != self.storage.length:
            raise ValueError(f"Native tuple carrier {role} length does not match template.")
        if not all(isinstance(item, Real) for item in value):
            raise TypeError(f"Native tuple carrier {role} must contain numeric values.")
        return cast(CarrierNativeTupleValue, value)
