from __future__ import annotations

from stark.engines.native.carriers.array.allocation import CarrierAllocationNativeArray
from stark.engines.native.carriers.array.basis import CarrierBasisNativeArray
from stark.engines.native.carriers.array.arithmetic import CarrierArithmeticNativeArray
from stark.engines.native.carriers.array.norm import CarrierNormNativeArrayRMS
from stark.engines.native.carriers.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.engines.native.carriers.array.validation import CarrierValidationNativeArray


class CarrierNativeArray:
    def __init__(self, template: CarrierNativeArrayValue) -> None:
        storage = CarrierStorageNativeArray.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeArray(storage)
        self.allocation = CarrierAllocationNativeArray(storage)
        self.basis = CarrierBasisNativeArray(storage)
        self.arithmetic = CarrierArithmeticNativeArray()
        self.norm = CarrierNormNativeArrayRMS()
