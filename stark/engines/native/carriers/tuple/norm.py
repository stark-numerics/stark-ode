from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from stark.engines.native.carriers.tuple.storage import CarrierNativeTupleValue


@dataclass(frozen=True)
class CarrierNormNativeTupleRMS:
    def __call__(self, value: CarrierNativeTupleValue) -> float:
        if not value:
            return 0.0
        return sqrt(sum(abs(item) ** 2 for item in value) / len(value))


@dataclass(frozen=True)
class CarrierNormNativeTupleMax:
    def __call__(self, value: CarrierNativeTupleValue) -> float:
        if not value:
            return 0.0
        return float(max(abs(item) for item in value))
