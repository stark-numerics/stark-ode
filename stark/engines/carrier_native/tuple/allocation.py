from __future__ import annotations

from dataclasses import dataclass

from stark.engines.carrier_native.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple


@dataclass(frozen=True)
class CarrierAllocationNativeTuple:
    storage: CarrierStorageNativeTuple

    def zero_state(self) -> CarrierNativeTupleValue:
        return tuple(0 for _ in range(self.storage.length))

    def zero_translation(self) -> CarrierNativeTupleValue:
        return self.zero_state()

    def allocate_translation(self) -> CarrierNativeTupleValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierNativeTupleValue) -> CarrierNativeTupleValue:
        return tuple(value)

    def copy_translation(self, value: CarrierNativeTupleValue) -> CarrierNativeTupleValue:
        return tuple(value)
