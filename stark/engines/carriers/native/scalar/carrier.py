from __future__ import annotations

from stark.engines.carriers.native.scalar.allocation import CarrierAllocationNativeScalar
from stark.engines.carriers.native.scalar.basis import CarrierBasisNativeScalar
from stark.engines.carriers.native.scalar.arithmetic import CarrierArithmeticNativeScalar
from stark.engines.carriers.native.scalar.norm import CarrierNormNativeScalarAbs
from stark.engines.carriers.native.scalar.storage import CarrierNativeScalarValue, CarrierStorageNativeScalar
from stark.engines.carriers.native.scalar.validation import CarrierValidationNativeScalar


class CarrierNativeScalar:
    def __init__(self, template: CarrierNativeScalarValue) -> None:
        storage = CarrierStorageNativeScalar(template)
        self.storage = storage
        self.validation = CarrierValidationNativeScalar(storage)
        self.allocation = CarrierAllocationNativeScalar(storage)
        self.basis = CarrierBasisNativeScalar(storage)
        self.arithmetic = CarrierArithmeticNativeScalar()
        self.norm = CarrierNormNativeScalarAbs()
