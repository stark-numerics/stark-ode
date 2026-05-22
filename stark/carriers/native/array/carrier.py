from __future__ import annotations

from stark.carriers.native.array.allocation import CarrierAllocationNativeArray
from stark.carriers.native.array.arithmetic import CarrierArithmeticNativeArray
from stark.carriers.native.array.norm import CarrierNormNativeArrayRMS
from stark.carriers.native.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.carriers.native.array.validation import CarrierValidationNativeArray


class CarrierNativeArray:
    def __init__(self, template: CarrierNativeArrayValue) -> None:
        storage = CarrierStorageNativeArray.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeArray(storage)
        self.allocation = CarrierAllocationNativeArray(storage)
        self.arithmetic = CarrierArithmeticNativeArray()
        self.norm = CarrierNormNativeArrayRMS()
