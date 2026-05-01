from __future__ import annotations

import numpy as np

from stark.accelerators import Accelerator
from stark.algebraist import Algebraist, AlgebraistField, AlgebraistLooped
from stark.carriers import CarrierNumpy
from stark.interface.vector import StarkVectorTranslation

from benchmarks.common import FPUT_SIZES


ARITIES = (1, 2, 4, 8, 12)
_ACCELERATED_CONTEXTS: dict[int, Algebraist] = {}


def numba_available() -> bool:
    try:
        Accelerator.numba(cache=False)
    except ModuleNotFoundError:
        return False
    return True


def coefficients(arity: int) -> tuple[float, ...]:
    return tuple((index + 1.0) / arity for index in range(arity))


def accelerated_algebraist():
    return Algebraist(
        fields=(
            AlgebraistField("dq", "q", policy=AlgebraistLooped(rank=1)),
            AlgebraistField("dp", "p", policy=AlgebraistLooped(rank=1)),
        ),
        accelerator=Accelerator.numba(cache=False),
        generate_norm="l2",
    )


def accelerated_context(size: int) -> Algebraist:
    algebraist = _ACCELERATED_CONTEXTS.get(size)
    if algebraist is None:
        algebraist = accelerated_algebraist()
        _ACCELERATED_CONTEXTS[size] = algebraist
    return algebraist


def vector_terms(size: int, arity: int) -> tuple[StarkVectorTranslation, ...]:
    carrier = CarrierNumpy().bind(np.zeros(2 * size, dtype=np.float64))
    grid = np.linspace(0.0, 1.0, 2 * size, dtype=np.float64)
    return tuple(
        StarkVectorTranslation(
            (index + 1.0) * grid.copy(),
            carrier,
        )
        for index in range(arity)
    )


class AcceleratedTranslation:
    __slots__ = ("dq", "dp", "linear_combine")

    def __init__(self, dq: np.ndarray, dp: np.ndarray, algebraist: Algebraist) -> None:
        self.dq = dq
        self.dp = dp
        self.linear_combine = algebraist.linear_combine


def accelerated_terms(
    size: int,
    arity: int,
    algebraist: Algebraist,
) -> tuple[AcceleratedTranslation, ...]:
    grid = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return tuple(
        AcceleratedTranslation(
            (index + 1.0) * grid.copy(),
            (index + 2.0) * grid.copy(),
            algebraist,
        )
        for index in range(arity)
    )


class TimeAlgebraistCombine:
    params = (FPUT_SIZES, ARITIES)
    param_names = ("chain_size", "arity")

    def setup(self, chain_size: int, arity: int) -> None:
        self.coefficients = coefficients(arity)
        self.vector_values = vector_terms(chain_size, arity)
        self.vector_out = StarkVectorTranslation(
            np.zeros(2 * chain_size, dtype=np.float64),
            self.vector_values[0].carrier,
            self.vector_values[0].routing,
        )
        self.vector_combine = self.vector_out.linear_combine[arity - 1]

    def time_vector_carrier_combine(self, chain_size: int, arity: int) -> None:
        del chain_size
        terms: list[object] = []
        for coefficient, value in zip(self.coefficients, self.vector_values, strict=True):
            terms.extend((coefficient, value))
        self.vector_combine(self.vector_out, *terms)


if numba_available():

    class TimeAcceleratedAlgebraistCombine:
        params = (FPUT_SIZES, ARITIES)
        param_names = ("chain_size", "arity")

        def setup(self, chain_size: int, arity: int) -> None:
            algebraist = accelerated_context(chain_size)
            self.coefficients = coefficients(arity)
            self.out = AcceleratedTranslation(
                np.zeros(chain_size, dtype=np.float64),
                np.zeros(chain_size, dtype=np.float64),
                algebraist,
            )
            self.values = accelerated_terms(chain_size, arity, algebraist)
            self.combine = self.out.linear_combine[arity - 1]
            self.terms: list[object] = []
            for coefficient, value in zip(self.coefficients, self.values, strict=True):
                self.terms.extend((coefficient, value))

            self.combine(self.out, *self.terms)

        def time_accelerated_algebraist_combine(self, chain_size: int, arity: int) -> None:
            del chain_size, arity
            self.combine(self.out, *self.terms)

    class TimeAcceleratedAlgebraistSetup:
        params = (FPUT_SIZES,)
        param_names = ("chain_size",)

        def time_accelerated_algebraist_compile(self, chain_size: int) -> None:
            algebraist = accelerated_algebraist()
            probe = np.zeros(chain_size, dtype=np.float64)
            algebraist.compile_examples(probe, probe)
