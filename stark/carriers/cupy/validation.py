from dataclasses import dataclass
from typing import Protocol, cast
import cupy as cp

from stark.carriers.cupy.storage import CarrierCupyValue, CarrierStorageCupy


class CupyModule(Protocol):
    ndarray: type[CarrierCupyValue]

    def asarray(self, value: CarrierCupyValue) -> CarrierCupyValue: ...

cupy = cast(CupyModule, cp)


@dataclass(frozen=True)
class CarrierValidationCupy:
    storage: CarrierStorageCupy

    def validate_state(self, value: CarrierCupyValue) -> CarrierCupyValue:
        return self.validate_array(value, "state")

    def validate_translation(self, value: CarrierCupyValue) -> CarrierCupyValue:
        return self.validate_array(value, "translation")

    def coerce_translation(self, value: CarrierCupyValue) -> CarrierCupyValue:
        return self.validate_translation(cupy.asarray(value))

    def validate_array(self, value: CarrierCupyValue, role: str) -> CarrierCupyValue:
        if not isinstance(value, cupy.ndarray):
            raise TypeError(f"Cupy carrier {role} must be a cupy.ndarray.")

        if not self.storage.matches_template(value):
            raise ValueError(f"Cupy carrier {role} shape does not match template.")

        return value