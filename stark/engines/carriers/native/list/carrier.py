from __future__ import annotations

from stark.engines.carriers.native.list.allocation import CarrierAllocationNativeList
from stark.engines.carriers.native.list.basis import CarrierBasisNativeList
from stark.engines.carriers.native.list.arithmetic import CarrierArithmeticNativeList
from stark.engines.carriers.native.list.norm import CarrierNormNativeListRMS
from stark.engines.carriers.native.list.storage import CarrierNativeListValue, CarrierStorageNativeList
from stark.engines.carriers.native.list.validation import CarrierValidationNativeList


class CarrierNativeList:
    def __init__(self, template: CarrierNativeListValue) -> None:
        storage = CarrierStorageNativeList.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeList(storage)
        self.allocation = CarrierAllocationNativeList(storage)
        self.basis = CarrierBasisNativeList(storage)
        self.arithmetic = CarrierArithmeticNativeList()
        self.norm = CarrierNormNativeListRMS()
