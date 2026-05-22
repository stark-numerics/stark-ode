from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from stark.carriers.native.list.storage import CarrierNativeListValue


@dataclass(frozen=True)
class CarrierNormNativeListRMS:
    def __call__(self, value: CarrierNativeListValue) -> float:
        if not value:
            return 0.0
        return sqrt(sum(abs(item) ** 2 for item in value) / len(value))


@dataclass(frozen=True)
class CarrierNormNativeListMax:
    def __call__(self, value: CarrierNativeListValue) -> float:
        if not value:
            return 0.0
        return float(max(abs(item) for item in value))
