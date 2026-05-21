from dataclasses import dataclass
from typing import Protocol, cast

import cupy as cp

from stark.carriers.cupy.storage import CarrierCupyValue


class CupyModule(Protocol):
    def abs(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def max(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def mean(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def sqrt(self, value: CarrierCupyValue) -> CarrierCupyValue: ...


cupy = cast(CupyModule, cp)


@dataclass(frozen=True)
class CarrierNormCupyRMS:
    def __call__(self, value: CarrierCupyValue) -> float:
        absolute = cupy.abs(value)
        return float(cupy.sqrt(cupy.mean(absolute ** 2)).item())


@dataclass(frozen=True)
class CarrierNormCupyMax:
    def __call__(self, value: CarrierCupyValue) -> float:
        return float(cupy.max(cupy.abs(value)).item())