from stark.carriers.native.allocation import CarrierAllocationNative
from stark.carriers.native.arithmetic import CarrierArithmeticNative
from stark.carriers.native.norm import CarrierNormNativeRMS
from stark.carriers.native.storage import CarrierNativeValue, CarrierStorageNative
from stark.carriers.native.validation import CarrierValidationNative


class CarrierNative:
    def __init__(self, template: CarrierNativeValue) -> None:
        storage = CarrierStorageNative(template)

        self.storage = storage
        self.validation = CarrierValidationNative(storage)
        self.allocation = CarrierAllocationNative(storage)
        self.arithmetic = CarrierArithmeticNative()
        self.norm = CarrierNormNativeRMS()
