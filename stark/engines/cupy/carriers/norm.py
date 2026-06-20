"""Norm policies for CuPy-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, cast

import cupy as cp

from stark.engines.cupy.carriers.storage import CarrierCupyValue


class HintCupyModule(Protocol):
    """Subset of CuPy reduction APIs used by carrier norms."""

    def abs(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def max(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def mean(self, value: CarrierCupyValue) -> CarrierCupyValue: ...
    def sqrt(self, value: CarrierCupyValue) -> CarrierCupyValue: ...


cupy = cast(HintCupyModule, cp)


@dataclass(frozen=True)
class CarrierNormCupyRMS:
    """Root-mean-square norm for CuPy arrays."""

    def __call__(self, value: CarrierCupyValue) -> float:
        absolute = cupy.abs(value)
        return float(cupy.sqrt(cupy.mean(absolute ** 2)).item())


@dataclass(frozen=True)
class CarrierNormCupyMax:
    """Maximum absolute-entry norm for CuPy arrays."""

    def __call__(self, value: CarrierCupyValue) -> float:
        return float(cupy.max(cupy.abs(value)).item())
