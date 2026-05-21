from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stark.carriers.numpy.storage import CarrierStorageNumpy


@dataclass(frozen=True)
class CarrierValidationNumpy:
    storage: CarrierStorageNumpy

    def validate_state(self, value: NDArray) -> NDArray:
        return self.validate_array(value, "state")

    def validate_translation(self, value: NDArray) -> NDArray:
        return self.validate_array(value, "translation")

    def coerce_translation(self, value: object) -> NDArray:
        return self.validate_translation(np.asarray(value))

    def validate_array(self, value: NDArray, role: str) -> NDArray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"NumPy carrier {role} must be a numpy.ndarray.")

        if not self.storage.matches_template(value):
            raise ValueError(f"NumPy carrier {role} shape does not match template.")

        return value