from typing import Any, cast

import numpy as np
from numpy.typing import DTypeLike

from stark.engines.carrier_numpy.allocation import CarrierAllocationNumpy
from stark.engines.carrier_numpy.basis import CarrierBasisNumpy
from stark.engines.carrier_numpy.arithmetic import CarrierArithmeticNumpy
from stark.engines.carrier_numpy.norm import CarrierNormNumpyRMS
from stark.engines.carrier_numpy.storage import CarrierNumpyValue, CarrierStorageNumpy
from stark.engines.carrier_numpy.validation import CarrierValidationNumpy
from stark.engines.carriers import CarrierScalarPython


class CarrierNumpy:
    @classmethod
    def from_shape(cls, shape: tuple[int, ...], dtype: object) -> "CarrierNumpy":
        return cls(np.zeros(shape, dtype=cast(DTypeLike, cls.resolve_dtype(dtype))))

    @staticmethod
    def resolve_dtype(dtype: object) -> np.dtype[Any]:
        return np.dtype(cast(DTypeLike, dtype))

    def __init__(self, template: CarrierNumpyValue) -> None:
        storage = CarrierStorageNumpy.from_template(template)

        self.storage = storage
        self.validation = CarrierValidationNumpy(storage)
        self.allocation = CarrierAllocationNumpy(storage)
        self.basis = CarrierBasisNumpy(storage)
        self.arithmetic = CarrierArithmeticNumpy(storage)
        self.norm = CarrierNormNumpyRMS()
        self.scalar = CarrierScalarPython()
