from __future__ import annotations

from array import array
from numbers import Real
from typing import Any, cast

from stark.engines.carrier_native.basis import CarrierBasisNative
from stark.engines.carrier_native.array import CarrierNativeArray
from stark.engines.carrier_native.hints import HintNativeNumber
from stark.engines.carrier_native.list import CarrierNativeList
from stark.engines.carrier_native.scalar import CarrierNativeScalar
from stark.engines.carrier_native.storage import CarrierNativeValue
from stark.engines.carrier_native.tuple import CarrierNativeTuple
from stark.engines.carriers import CarrierScalarPython


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
        self.scalar = CarrierScalarPython()

    @staticmethod
    def _select(template: CarrierNativeValue) -> Any:
        if isinstance(template, Real):
            return CarrierNativeScalar(cast(HintNativeNumber, template))
        if isinstance(template, list):
            return CarrierNativeList(template)
        if isinstance(template, tuple):
            return CarrierNativeTuple(template)
        if isinstance(template, array):
            return CarrierNativeArray(template)
        raise TypeError(
            "Native carrier template must be numeric, list, tuple, or floating array.array."
        )
