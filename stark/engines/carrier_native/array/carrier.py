from __future__ import annotations

from stark.engines.carrier_native.array.allocation import CarrierAllocationNativeArray
from stark.engines.carrier_native.array.basis import CarrierBasisNativeArray
from stark.engines.carrier_native.array.arithmetic import CarrierArithmeticNativeArray
from stark.engines.carrier_native.array.norm import CarrierNormNativeArrayRMS
from stark.engines.carrier_native.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.engines.carrier_native.array.validation import CarrierValidationNativeArray
from stark.engines.carriers import CarrierScalarPython


class CarrierNativeArray:
    def __init__(self, template: CarrierNativeArrayValue) -> None:
        storage = CarrierStorageNativeArray.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeArray(storage)
        self.allocation = CarrierAllocationNativeArray(storage)
        self.basis = CarrierBasisNativeArray(storage)
        self.arithmetic = CarrierArithmeticNativeArray()
        self.norm = CarrierNormNativeArrayRMS()
        self.scalar = CarrierScalarPython()
