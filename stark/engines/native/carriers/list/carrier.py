from __future__ import annotations

from stark.engines.native.carriers.list.allocation import CarrierAllocationNativeList
from stark.engines.native.carriers.list.basis import CarrierBasisNativeList
from stark.engines.native.carriers.list.arithmetic import CarrierArithmeticNativeList
from stark.engines.native.carriers.list.norm import CarrierNormNativeListRMS
from stark.engines.native.carriers.list.storage import CarrierNativeListValue, CarrierStorageNativeList
from stark.engines.native.carriers.list.validation import CarrierValidationNativeList


class CarrierNativeList:
    def __init__(self, template: CarrierNativeListValue) -> None:
        storage = CarrierStorageNativeList.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeList(storage)
        self.allocation = CarrierAllocationNativeList(storage)
        self.basis = CarrierBasisNativeList(storage)
        self.arithmetic = CarrierArithmeticNativeList()
        self.norm = CarrierNormNativeListRMS()
