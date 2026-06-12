from __future__ import annotations

from dataclasses import dataclass

from stark.engines.carriers.native.scalar.storage import (
    CarrierNativeScalarValue,
    CarrierStorageNativeScalar,
)


@dataclass(frozen=True)
class CarrierAllocationNativeScalar:
    storage: CarrierStorageNativeScalar

    def zero_state(self) -> CarrierNativeScalarValue:
        return 0

    def zero_translation(self) -> CarrierNativeScalarValue:
        return 0

    def allocate_translation(self) -> CarrierNativeScalarValue:
        return 0

    def copy_state(self, value: CarrierNativeScalarValue) -> CarrierNativeScalarValue:
        return value

    def copy_translation(self, value: CarrierNativeScalarValue) -> CarrierNativeScalarValue:
        return value
