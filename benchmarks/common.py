from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.generator import AlgebraistGeneratorLinearCombine
from stark.engines.algebraist.layout import AlgebraistLayout, AlgebraistLayoutField, AlgebraistLayoutLooped


FPUT_SIZES = (64, 512, 2048)


@dataclass(frozen=True, slots=True)
class FPUTParameters:
    chain_size: int
    t0: float = 0.0
    t1: float = 20.0
    initial_step: float = 1.0e-3
    beta: float = 0.25
    amplitude: float = 0.1


FPUT_ALGEBRAIST_LAYOUT = AlgebraistLayout(
    fields=(
        AlgebraistLayoutField("dq", "q", policy=AlgebraistLayoutLooped(rank=1)),
        AlgebraistLayoutField("dp", "p", policy=AlgebraistLayoutLooped(rank=1)),
    ),
)


class FPUTState:
    __slots__ = ("q", "p")

    def __init__(self, q: np.ndarray, p: np.ndarray) -> None:
        self.q = q
        self.p = p

    def copy(self) -> "FPUTState":
        return FPUTState(self.q.copy(), self.p.copy())


class FPUTTranslation:
    __slots__ = ("dq", "dp")

    def __init__(self, dq: np.ndarray, dp: np.ndarray) -> None:
        self.dq = dq
        self.dp = dp

    def __call__(self, origin: FPUTState, result: FPUTState) -> None:
        result.q[...] = origin.q + self.dq
        result.p[...] = origin.p + self.dp

    def norm(self) -> float:
        energy = np.dot(self.dq.ravel(), self.dq.ravel()) + np.dot(self.dp.ravel(), self.dp.ravel())
        return sqrt(float(energy) / self.dq.size)

    def __add__(self, other: "FPUTTranslation") -> "FPUTTranslation":
        return FPUTTranslation(self.dq + other.dq, self.dp + other.dp)

    def __rmul__(self, scalar: float) -> "FPUTTranslation":
        return FPUTTranslation(scalar * self.dq, scalar * self.dp)


class FPUTAllocator:
    __slots__ = ("chain_size",)

    def __init__(self, parameters: FPUTParameters) -> None:
        self.chain_size = parameters.chain_size
        if not hasattr(FPUTTranslation, "linear_combine"):
            FPUTTranslation.linear_combine = _generated_fput_linear_combine(self)

    def allocate_state(self) -> FPUTState:
        return FPUTState(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )

    def copy_state(self, source: FPUTState, out: FPUTState) -> None:
        np.copyto(out.q, source.q)
        np.copyto(out.p, source.p)

    def allocate_translation(self) -> FPUTTranslation:
        return FPUTTranslation(
            np.zeros(self.chain_size, dtype=np.float64),
            np.zeros(self.chain_size, dtype=np.float64),
        )


def _generated_fput_linear_combine(allocator: FPUTAllocator):
    provider = AlgebraistGeneratorLinearCombine(
        translation=allocator.allocate_translation(),
        allocator=allocator,
        layout=FPUT_ALGEBRAIST_LAYOUT,
    )
    return tuple(provider.provide(AlgebraistArity(arity)) for arity in range(1, 13))


class FPUTDerivative:
    __slots__ = ("beta", "left", "right")

    def __init__(self, parameters: FPUTParameters) -> None:
        self.beta = parameters.beta
        self.left = np.zeros(parameters.chain_size, dtype=np.float64)
        self.right = np.zeros(parameters.chain_size, dtype=np.float64)

    def __call__(self, interval, state: FPUTState, out: FPUTTranslation) -> None:
        del interval
        q = state.q
        left = self.left
        right = self.right

        left[0] = 0.0
        left[1:] = q[:-1]
        right[-1] = 0.0
        right[:-1] = q[1:]

        out.dq[:] = state.p
        out.dp[:] = right - 2.0 * q + left + self.beta * ((right - q) ** 3 - (q - left) ** 3)


class FPUTVectorDerivative:
    __slots__ = ("beta", "chain_size", "left", "right")

    def __init__(self, parameters: FPUTParameters) -> None:
        self.beta = parameters.beta
        self.chain_size = parameters.chain_size
        self.left = np.zeros(parameters.chain_size, dtype=np.float64)
        self.right = np.zeros(parameters.chain_size, dtype=np.float64)

    def __call__(self, t: float, y: np.ndarray, dy: np.ndarray) -> None:
        del t
        size = self.chain_size
        q = y[:size]
        p = y[size:]
        dq = dy[:size]
        dp = dy[size:]
        left = self.left
        right = self.right

        left[0] = 0.0
        left[1:] = q[:-1]
        right[-1] = 0.0
        right[:-1] = q[1:]

        dq[:] = p
        dp[:] = right - 2.0 * q + left + self.beta * ((right - q) ** 3 - (q - left) ** 3)


class FPUTVectorReturnDerivative:
    __slots__ = ("in_place", "scratch")

    def __init__(self, parameters: FPUTParameters) -> None:
        self.in_place = FPUTVectorDerivative(parameters)
        self.scratch = np.zeros(2 * parameters.chain_size, dtype=np.float64)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        out = self.scratch.copy()
        self.in_place(t, y, out)
        return out


class FPUTMatrixDerivative:
    __slots__ = ("beta", "chain_size", "left", "right")

    def __init__(self, parameters: FPUTParameters) -> None:
        self.beta = parameters.beta
        self.chain_size = parameters.chain_size
        self.left = np.zeros(parameters.chain_size, dtype=np.float64)
        self.right = np.zeros(parameters.chain_size, dtype=np.float64)

    def __call__(self, t: float, y: np.ndarray, dy: np.ndarray) -> None:
        del t
        q = y[0]
        p = y[1]
        dq = dy[0]
        dp = dy[1]
        left = self.left
        right = self.right

        left[0] = 0.0
        left[1:] = q[:-1]
        right[-1] = 0.0
        right[:-1] = q[1:]

        dq[:] = p
        dp[:] = right - 2.0 * q + left + self.beta * ((right - q) ** 3 - (q - left) ** 3)


class FPUTMatrixReturnDerivative:
    __slots__ = ("in_place", "scratch")

    def __init__(self, parameters: FPUTParameters) -> None:
        self.in_place = FPUTMatrixDerivative(parameters)
        self.scratch = np.zeros((2, parameters.chain_size), dtype=np.float64)

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        out = self.scratch.copy()
        self.in_place(t, y, out)
        return out


def fput_parameters(chain_size: int) -> FPUTParameters:
    return FPUTParameters(chain_size=chain_size)


def initial_fput_state(parameters: FPUTParameters) -> FPUTState:
    indices = np.arange(1, parameters.chain_size + 1, dtype=np.float64)
    q = parameters.amplitude * np.sin(np.pi * indices / (parameters.chain_size + 1))
    p = np.zeros(parameters.chain_size, dtype=np.float64)
    return FPUTState(q.astype(np.float64), p)


def initial_fput_vector(parameters: FPUTParameters) -> np.ndarray:
    state = initial_fput_state(parameters)
    return np.concatenate((state.q, state.p))


def initial_fput_matrix(parameters: FPUTParameters) -> np.ndarray:
    state = initial_fput_state(parameters)
    return np.stack((state.q, state.p))


def fput_problem(chain_size: int) -> tuple[FPUTParameters, FPUTState, FPUTDerivative, FPUTAllocator]:
    parameters = fput_parameters(chain_size)
    return (
        parameters,
        initial_fput_state(parameters),
        FPUTDerivative(parameters),
        FPUTAllocator(parameters),
    )


def fput_vector_problem(chain_size: int) -> tuple[FPUTParameters, np.ndarray, FPUTVectorDerivative]:
    parameters = fput_parameters(chain_size)
    return (
        parameters,
        initial_fput_vector(parameters),
        FPUTVectorDerivative(parameters),
    )


def fput_matrix_problem(chain_size: int) -> tuple[FPUTParameters, np.ndarray, FPUTMatrixDerivative]:
    parameters = fput_parameters(chain_size)
    return (
        parameters,
        initial_fput_matrix(parameters),
        FPUTMatrixDerivative(parameters),
    )
