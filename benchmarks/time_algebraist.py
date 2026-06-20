from __future__ import annotations

import numpy as np

from stark.engines.accelerators import AcceleratorNumba
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator import AlgebraistGeneratorLinearCombine
from stark.engines.algebraist.frame import AlgebraistFrame, AlgebraistFrameField, AlgebraistFrameLooped
from stark.engines.numpy import EngineNumpy
from stark.problem.frame.frame import Frame

from benchmarks.common import FPUT_SIZES


ARITIES = (1, 2, 4, 8, 12)
_ACCELERATED_CONTEXTS: dict[int, tuple] = {}


def numba_available() -> bool:
    try:
        AcceleratorNumba(cache=False)
    except ModuleNotFoundError:
        return False
    return True


def coefficients(arity: int) -> tuple[float, ...]:
    return tuple((index + 1.0) / arity for index in range(arity))


def accelerated_layout() -> AlgebraistFrame:
    return AlgebraistFrame(
        fields=(
            AlgebraistFrameField("dq", "q", policy=AlgebraistFrameLooped(rank=1)),
            AlgebraistFrameField("dp", "p", policy=AlgebraistFrameLooped(rank=1)),
        ),
    )


def accelerated_context(size: int) -> tuple:
    linear_combine = _ACCELERATED_CONTEXTS.get(size)
    if linear_combine is None:
        allocator = AcceleratedAllocator(size)
        provider = AlgebraistGeneratorLinearCombine(
            translation=allocator.allocate_translation(),
            allocator=allocator,
            frame=accelerated_layout(),
            accelerator=AcceleratorNumba(cache=False),
        )
        linear_combine = tuple(provider.provide(AlgebraistArity(arity)) for arity in range(1, 13))
        _ACCELERATED_CONTEXTS[size] = linear_combine
    return linear_combine


def engine_terms(size: int, arity: int) -> tuple:
    engine = EngineNumpy(
        Frame({"y": {"translation": "dy", "shape": (2 * size,)}}),
    )
    grid = np.linspace(0.0, 1.0, 2 * size, dtype=np.float64)
    terms = []
    for index in range(arity):
        term = engine.allocator.allocate_translation()
        term.dy[...] = (index + 1.0) * grid
        terms.append(term)
    return tuple(terms)


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
        self.engine_values = engine_terms(chain_size, arity)
        self.engine_out = self.engine_values[0].allocator.allocate_translation()
        self.engine_combine = self.engine_out.linear_combine[arity - 1]

    def time_engine_combine(self, chain_size: int, arity: int) -> None:
        del chain_size
        terms: list[object] = []
        for coefficient, value in zip(self.coefficients, self.engine_values, strict=True):
            terms.extend((coefficient, value))
        self.engine_combine(*terms, self.engine_out)


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
            provider = AlgebraistGeneratorLinearCombine(
                translation=allocator.allocate_translation(),
                allocator=allocator,
                frame=accelerated_layout(),
                accelerator=AcceleratorNumba(cache=False),
            )
            for arity in ARITIES:
                provider.provide(AlgebraistArity(arity))
