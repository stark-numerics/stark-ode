from __future__ import annotations

from array import array
from numbers import Number
from typing import Any

from stark.engines.carriers.native.basis import CarrierBasisNative
from stark.engines.carriers.native.array import CarrierNativeArray
from stark.engines.carriers.native.list import CarrierNativeList
from stark.engines.carriers.native.scalar import CarrierNativeScalar
from stark.engines.carriers.native.storage import CarrierNativeValue
from stark.engines.carriers.native.tuple import CarrierNativeTuple


class CarrierNative:
    """Facade that binds a concrete native carrier once from the template type."""

    concrete: Any

    def __init__(self, template: CarrierNativeValue) -> None:
        concrete = self._select(template)
        self.concrete = concrete
        self.storage = concrete.storage
        self.validation = concrete.validation
        self.allocation = concrete.allocation
        self.basis = CarrierBasisNative(concrete.storage)
        self.arithmetic = concrete.arithmetic
        self.norm = concrete.norm

    @staticmethod
    def _select(template: CarrierNativeValue) -> Any:
        if isinstance(template, Number):
            return CarrierNativeScalar(template)
        if isinstance(template, list):
            return CarrierNativeList(template)
        if isinstance(template, tuple):
            return CarrierNativeTuple(template)
        if isinstance(template, array):
            return CarrierNativeArray(template)
        raise TypeError(
            "Native carrier template must be numeric, list, tuple, or floating array.array."
        )
