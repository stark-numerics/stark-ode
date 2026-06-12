from __future__ import annotations

from array import array
from dataclasses import dataclass
from math import sqrt
from numbers import Number

from stark.engines.carriers.native.storage import CarrierNativeValue


@dataclass(frozen=True)
class CarrierNormNativeScalarAbs:
    def __call__(self, value: CarrierNativeValue) -> float:
        if not isinstance(value, Number):
            raise TypeError("Native scalar norm requires a numeric value.")
        return float(abs(value))


@dataclass(frozen=True)
class CarrierNormNativeRMS:
    def __call__(self, value: CarrierNativeValue) -> float:
        if isinstance(value, Number):
            return float(abs(value))
        if isinstance(value, array | list | tuple):
            if not value:
                return 0.0
            return sqrt(sum(abs(item) ** 2 for item in value) / len(value))
        raise TypeError("Native RMS norm requires numeric, list, tuple, or array.array value.")


@dataclass(frozen=True)
class CarrierNormNativeMax:
    def __call__(self, value: CarrierNativeValue) -> float:
        if isinstance(value, Number):
            return float(abs(value))
        if isinstance(value, array | list | tuple):
            if not value:
                return 0.0
            return float(max(abs(item) for item in value))
        raise TypeError("Native max norm requires numeric, list, tuple, or array.array value.")
