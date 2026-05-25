from dataclasses import dataclass
from typing import Protocol, cast

import cupy as cp

from stark.carriers.cupy.storage import CarrierCupyValue, CarrierStorageCupy


class HintCupyModule(Protocol):
    def zeros(self, shape: tuple[int, ...], dtype: object) -> CarrierCupyValue: ...
    def array(self, value: CarrierCupyValue, *, copy: bool = ...) -> CarrierCupyValue: ...


cupy = cast(HintCupyModule, cp)


@dataclass(frozen=True)
class CarrierAllocationCupy:
    storage: CarrierStorageCupy

    def zero_state(self) -> CarrierCupyValue:
        return cupy.zeros(self.storage.shape, dtype=self.storage.dtype)

    def zero_translation(self) -> CarrierCupyValue:
        return self.zero_state()

    def allocate_translation(self) -> CarrierCupyValue:
        return self.zero_translation()

    def copy_state(self, value: CarrierCupyValue) -> CarrierCupyValue:
        return cupy.array(value, copy=True)

    def copy_translation(self, value: CarrierCupyValue) -> CarrierCupyValue:
        return self.copy_state(value)