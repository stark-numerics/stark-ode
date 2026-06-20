"""Storage checks for CuPy-backed carrier values."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import cupy as cp


class HintCupyArray(Protocol):
    """Small structural type for the CuPy array attributes STARK uses."""

    shape: tuple[int, ...]
    dtype: object


class HintCupyModule(Protocol):
    """Subset of the CuPy module used by carrier storage."""

    ndarray: type[HintCupyArray]

    def asarray(self, value: HintCupyArray) -> HintCupyArray: ...


cupy = cast(HintCupyModule, cp)
CarrierCupyValue: TypeAlias = HintCupyArray


@dataclass(frozen=True)
class CarrierStorageCupy:
    """Template metadata for CuPy states and translations."""

    shape: tuple[int, ...]
    dtype: object

    @classmethod
    def from_template(cls, template: CarrierCupyValue) -> "CarrierStorageCupy":
        array = cupy.asarray(template)
        return cls(shape=array.shape, dtype=array.dtype)

    def is_state(self, value: CarrierCupyValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierCupyValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: CarrierCupyValue) -> bool:
        return isinstance(value, cupy.ndarray) and value.shape == self.shape
