from __future__ import annotations

from stark.engines.carriers.native.array.allocation import CarrierAllocationNativeArray
from stark.engines.carriers.native.array.basis import CarrierBasisNativeArray
from stark.engines.carriers.native.array.arithmetic import CarrierArithmeticNativeArray
from stark.engines.carriers.native.array.norm import CarrierNormNativeArrayRMS
from stark.engines.carriers.native.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.engines.carriers.native.array.validation import CarrierValidationNativeArray


class CarrierNativeArray:
    def __init__(self, template: CarrierNativeArrayValue) -> None:
        storage = CarrierStorageNativeArray.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeArray(storage)
        self.allocation = CarrierAllocationNativeArray(storage)
        self.basis = CarrierBasisNativeArray(storage)
        self.arithmetic = CarrierArithmeticNativeArray()
        self.norm = CarrierNormNativeArrayRMS()
