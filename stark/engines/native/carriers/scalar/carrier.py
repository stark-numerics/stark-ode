from __future__ import annotations

from stark.engines.native.carriers.scalar.allocation import CarrierAllocationNativeScalar
from stark.engines.native.carriers.scalar.basis import CarrierBasisNativeScalar
from stark.engines.native.carriers.scalar.arithmetic import CarrierArithmeticNativeScalar
from stark.engines.native.carriers.scalar.norm import CarrierNormNativeScalarAbs
from stark.engines.native.carriers.scalar.storage import CarrierNativeScalarValue, CarrierStorageNativeScalar
from stark.engines.native.carriers.scalar.validation import CarrierValidationNativeScalar


class CarrierNativeScalar:
    def __init__(self, template: CarrierNativeScalarValue) -> None:
        storage = CarrierStorageNativeScalar(template)
        self.storage = storage
        self.validation = CarrierValidationNativeScalar(storage)
        self.allocation = CarrierAllocationNativeScalar(storage)
        self.basis = CarrierBasisNativeScalar(storage)
        self.arithmetic = CarrierArithmeticNativeScalar()
        self.norm = CarrierNormNativeScalarAbs()
