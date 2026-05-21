from dataclasses import dataclass
from typing import Protocol, TypeAlias, cast

import cupy as cp

class CupyArray(Protocol):
    shape: tuple[int, ...]
    dtype: object

class CupyModule(Protocol):
    ndarray: type[CupyArray]
    def asarray(self, value: CupyArray) -> CupyArray: ...

cupy = cast(CupyModule, cp)
CarrierCupyValue: TypeAlias = CupyArray


@dataclass(frozen=True)
class CarrierStorageCupy:
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