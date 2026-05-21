from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CarrierArithmeticNumpy:
    preference: Literal["into"] = "into"

    def translate(self, state: NDArray, step: float, derivative: NDArray, result: NDArray) -> None:
        np.multiply(derivative, step, out=result)
        np.add(state, result, out=result)

    def add(self, left: NDArray, right: NDArray, result: NDArray) -> None:
        np.add(left, right, out=result)

    def scale(self, factor: float, value: NDArray, result: NDArray) -> None:
        np.multiply(value, factor, out=result)

    def combine(self, terms: tuple[tuple[float, NDArray], ...], result: NDArray) -> None:
        result[...] = 0
        for factor, value in terms:
            result[...] += factor * value