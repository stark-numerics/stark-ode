from __future__ import annotations

from stark.carriers.native.tuple.allocation import CarrierAllocationNativeTuple
from stark.carriers.native.tuple.arithmetic import CarrierArithmeticNativeTuple
from stark.carriers.native.tuple.norm import CarrierNormNativeTupleRMS
from stark.carriers.native.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple
from stark.carriers.native.tuple.validation import CarrierValidationNativeTuple


class CarrierNativeTuple:
    def __init__(self, template: CarrierNativeTupleValue) -> None:
        storage = CarrierStorageNativeTuple.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeTuple(storage)
        self.allocation = CarrierAllocationNativeTuple(storage)
        self.arithmetic = CarrierArithmeticNativeTuple()
        self.norm = CarrierNormNativeTupleRMS()
