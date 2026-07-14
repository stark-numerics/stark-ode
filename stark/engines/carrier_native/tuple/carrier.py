from __future__ import annotations

from stark.engines.carrier_native.tuple.allocation import CarrierAllocationNativeTuple
from stark.engines.carrier_native.tuple.basis import CarrierBasisNativeTuple
from stark.engines.carrier_native.tuple.arithmetic import CarrierArithmeticNativeTuple
from stark.engines.carrier_native.tuple.norm import CarrierNormNativeTupleRMS
from stark.engines.carrier_native.tuple.storage import CarrierNativeTupleValue, CarrierStorageNativeTuple
from stark.engines.carrier_native.tuple.validation import CarrierValidationNativeTuple
from stark.engines.carriers import CarrierScalarPython


class CarrierNativeTuple:
    def __init__(self, template: CarrierNativeTupleValue) -> None:
        storage = CarrierStorageNativeTuple.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeTuple(storage)
        self.allocation = CarrierAllocationNativeTuple(storage)
        self.basis = CarrierBasisNativeTuple(storage)
        self.arithmetic = CarrierArithmeticNativeTuple()
        self.norm = CarrierNormNativeTupleRMS()
        self.scalar = CarrierScalarPython()
