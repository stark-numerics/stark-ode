from __future__ import annotations

from array import array

from stark.engines.carrier_native.array.allocation import CarrierAllocationNativeArray
from stark.engines.carrier_native.array.basis import CarrierBasisNativeArray
from stark.engines.carrier_native.array.arithmetic import CarrierArithmeticNativeArray
from stark.engines.carrier_native.array.norm import CarrierNormNativeArrayRMS
from stark.engines.carrier_native.array.storage import CarrierNativeArrayValue, CarrierStorageNativeArray
from stark.engines.carrier_native.array.validation import CarrierValidationNativeArray
from stark.engines.carriers import CarrierScalarPython


class CarrierNativeArray:
    @classmethod
    def from_shape(
        cls,
        shape: tuple[int, ...],
        dtype: object = "d",
    ) -> "CarrierNativeArray":
        if len(shape) != 1:
            raise ValueError(
                "CarrierNativeArray.from_shape currently supports "
                "one-dimensional field shapes only."
            )
        typecode = cls.resolve_dtype(dtype)
        return cls(array(typecode, (0.0 for _ in range(shape[0]))))

    @staticmethod
    def resolve_dtype(dtype: object = "d") -> str:
        if not isinstance(dtype, str):
            raise TypeError(
                "CarrierNativeArray.from_shape expects an array.array typecode "
                f"string; got {type(dtype).__name__}."
            )
        return dtype

    def __init__(self, template: CarrierNativeArrayValue) -> None:
        storage = CarrierStorageNativeArray.from_template(template)
        self.storage = storage
        self.validation = CarrierValidationNativeArray(storage)
        self.allocation = CarrierAllocationNativeArray(storage)
        self.basis = CarrierBasisNativeArray(storage)
        self.arithmetic = CarrierArithmeticNativeArray()
        self.norm = CarrierNormNativeArrayRMS()
        self.scalar = CarrierScalarPython()
