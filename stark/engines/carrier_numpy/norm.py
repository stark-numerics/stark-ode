from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CarrierNormNumpyRMS:
    def __call__(self, value: NDArray) -> float:
        return float(np.sqrt(np.mean(np.abs(value) ** 2)))


@dataclass(frozen=True)
class CarrierNormNumpyMax:
    def __call__(self, value: NDArray) -> float:
        return float(np.max(np.abs(value)))