from __future__ import annotations

from stark.engines.carrier_native.scalar.allocation import CarrierAllocationNativeScalar
from stark.engines.carrier_native.scalar.basis import CarrierBasisNativeScalar
from stark.engines.carrier_native.scalar.arithmetic import CarrierArithmeticNativeScalar
from stark.engines.carrier_native.scalar.norm import CarrierNormNativeScalarAbs
from stark.engines.carrier_native.scalar.storage import CarrierNativeScalarValue, CarrierStorageNativeScalar
from stark.engines.carrier_native.scalar.validation import CarrierValidationNativeScalar
from stark.engines.carriers import CarrierScalarPython


class CarrierNativeScalar:
    def __init__(self, template: CarrierNativeScalarValue) -> None:
        storage = CarrierStorageNativeScalar(template)
        self.storage = storage
        self.validation = CarrierValidationNativeScalar(storage)
        self.allocation = CarrierAllocationNativeScalar(storage)
        self.basis = CarrierBasisNativeScalar(storage)
        self.arithmetic = CarrierArithmeticNativeScalar()
        self.norm = CarrierNormNativeScalarAbs()
        self.scalar = CarrierScalarPython()
