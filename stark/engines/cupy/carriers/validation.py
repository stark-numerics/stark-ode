"""Validation and coercion for CuPy-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, cast

import cupy as cp

from stark.engines.cupy.carriers.storage import CarrierCupyValue, CarrierStorageCupy


class HintCupyModule(Protocol):
    """Subset of CuPy validation APIs used by this carrier."""

    ndarray: type[CarrierCupyValue]

    def asarray(self, value: CarrierCupyValue) -> CarrierCupyValue: ...


cupy = cast(HintCupyModule, cp)


@dataclass(frozen=True)
class CarrierValidationCupy:
    """Validate that CuPy values match the carrier template."""

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
