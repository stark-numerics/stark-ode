from __future__ import annotations

from stark.engines.native.carriers.tuple.allocation import CarrierAllocationNativeTuple
from stark.engines.native.carriers.tuple.basis import CarrierBasisNativeTuple
from stark.engines.native.carriers.tuple.arithmetic import CarrierArithmeticNativeTuple
from stark.engines.native.carriers.tuple.norm import CarrierNormNativeTupleRMS
from stark.engines.native.carriers.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple
from stark.engines.native.carriers.tuple.validation import CarrierValidationNativeTuple


class CarrierNativeTuple:
    def __init__(self, template: CarrierNativeTupleValue) -> None:
        storage = CarrierStorageNativeTuple.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeTuple(storage)
        self.allocation = CarrierAllocationNativeTuple(storage)
        self.basis = CarrierBasisNativeTuple(storage)
        self.arithmetic = CarrierArithmeticNativeTuple()
        self.norm = CarrierNormNativeTupleRMS()
