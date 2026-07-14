from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

CarrierNumpyValue: TypeAlias = NDArray


@dataclass(frozen=True)
class CarrierStorageNumpy:
    shape: tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def from_template(cls, template: CarrierNumpyValue) -> "CarrierStorageNumpy":
        array = np.asarray(template)
        return cls(shape=array.shape, dtype=array.dtype)

    def is_state(self, value: CarrierNumpyValue) -> bool:
        return self.matches_template(value)

    def is_translation(self, value: CarrierNumpyValue) -> bool:
        return self.matches_template(value)

    def matches_template(self, value: CarrierNumpyValue) -> bool:
        if not isinstance(value, np.ndarray):
            return False

        return value.shape == self.shape