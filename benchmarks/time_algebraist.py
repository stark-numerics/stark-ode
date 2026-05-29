from __future__ import annotations

import numpy as np

from stark.accelerators import Accelerator
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.generator import AlgebraistGeneratorGeneral
from stark.algebraist.layout import AlgebraistLayout, AlgebraistLayoutField, AlgebraistLayoutLooped
from stark.carriers import CarrierNumpy
from stark.interface.vector import StarkVectorTranslation

from benchmarks.common import FPUT_SIZES


ARITIES = (1, 2, 4, 8, 12)
_ACCELERATED_CONTEXTS: dict[int, tuple] = {}


def numba_available() -> bool:
    try:
        Accelerator.numba(cache=False)
    except ModuleNotFoundError:
        return False
    return True


def coefficients(arity: int) -> tuple[float, ...]:
    return tuple((index + 1.0) / arity for index in range(arity))


def accelerated_layout() -> AlgebraistLayout:
    return AlgebraistLayout(
        fields=(
            AlgebraistLayoutField("dq", "q", policy=AlgebraistLayoutLooped(rank=1)),
            AlgebraistLayoutField("dp", "p", policy=AlgebraistLayoutLooped(rank=1)),
        ),
    )


def accelerated_context(size: int) -> tuple:
    linear_combine = _ACCELERATED_CONTEXTS.get(size)
    if linear_combine is None:
        allocator = AcceleratedAllocator(size)
        provider = AlgebraistGeneratorGeneral(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            layout=accelerated_layout(),
            accelerator=Accelerator.numba(cache=False),
        )
        linear_combine = tuple(provider.provide(AlgebraistArity(arity)) for arity in range(1, 13))
        _ACCELERATED_CONTEXTS[size] = linear_combine
    return linear_combine


def vector_terms(size: int, arity: int) -> tuple[StarkVectorTranslation, ...]:
    carrier = CarrierNumpy(np.zeros(2 * size, dtype=np.float64))
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

    def __init__(self, dq: np.ndarray, dp: np.ndarray, linear_combine: tuple) -> None:
        self.dq = dq
        self.dp = dp
        self.linear_combine = linear_combine


class AcceleratedAllocator:
    __slots__ = ("linear_combine", "size")

    def __init__(self, size: int) -> None:
        self.size = size
        self.linear_combine: tuple = ()

    def allocate_translation(self) -> AcceleratedTranslation:
        return AcceleratedTranslation(
            np.zeros(self.size, dtype=np.float64),
            np.zeros(self.size, dtype=np.float64),
            self.linear_combine,
        )


def accelerated_terms(
    size: int,
    arity: int,
    linear_combine: tuple,
) -> tuple[AcceleratedTranslation, ...]:
    grid = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return tuple(
        AcceleratedTranslation(
            (index + 1.0) * grid.copy(),
            (index + 2.0) * grid.copy(),
            linear_combine,
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
        )
        self.vector_combine = self.vector_out.linear_combine[arity - 1]

    def time_vector_carrier_combine(self, chain_size: int, arity: int) -> None:
        del chain_size
        terms: list[object] = []
        for coefficient, value in zip(self.coefficients, self.vector_values, strict=True):
            terms.extend((coefficient, value))
        self.vector_combine(*terms, self.vector_out)


if numba_available():

    class TimeAcceleratedAlgebraistCombine:
        params = (FPUT_SIZES, ARITIES)
        param_names = ("chain_size", "arity")

        def setup(self, chain_size: int, arity: int) -> None:
            linear_combine = accelerated_context(chain_size)
            self.coefficients = coefficients(arity)
            self.out = AcceleratedTranslation(
                np.zeros(chain_size, dtype=np.float64),
                np.zeros(chain_size, dtype=np.float64),
                linear_combine,
            )
            self.values = accelerated_terms(chain_size, arity, linear_combine)
            self.combine = self.out.linear_combine[arity - 1]
            self.terms: list[object] = []
            for coefficient, value in zip(self.coefficients, self.values, strict=True):
                self.terms.extend((coefficient, value))

            self.combine(*self.terms, self.out)

        def time_accelerated_algebraist_combine(self, chain_size: int, arity: int) -> None:
            del chain_size, arity
            self.combine(*self.terms, self.out)

    class TimeAcceleratedAlgebraistSetup:
        params = (FPUT_SIZES,)
        param_names = ("chain_size",)

        def time_accelerated_algebraist_compile(self, chain_size: int) -> None:
            allocator = AcceleratedAllocator(chain_size)
            provider = AlgebraistGeneratorGeneral(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                layout=accelerated_layout(),
                accelerator=Accelerator.numba(cache=False),
            )
            for arity in ARITIES:
                provider.provide(AlgebraistArity(arity))
