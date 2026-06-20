from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stark.engines.numpy.carriers.storage import CarrierStorageNumpy


@dataclass(frozen=True)
class CarrierAllocationNumpy:
    storage: CarrierStorageNumpy

    def zero_state(self) -> NDArray:
        return np.zeros(self.storage.shape, dtype=self.storage.dtype)

    def zero_translation(self) -> NDArray:
        return self.zero_state()

    def allocate_translation(self) -> NDArray:
        return self.zero_translation()

    def copy_state(self, value: NDArray) -> NDArray:
        return np.array(value, copy=True)

    def copy_translation(self, value: NDArray) -> NDArray:
        return self.copy_state(value)