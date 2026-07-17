"""Shared typed test fixtures for STARK's public contracts.

These helpers are intentionally tiny but complete. They satisfy the same
state, translation, allocator, dynamics, and fixed-linear contracts that user
code is expected to satisfy, which keeps tests focused on the behaviour under
test instead of on half-real local fakes.

The scalar fixtures are useful for scheme, resolvent, and inverter tests that
only need a one-dimensional state space. They make Pyright complaints more
meaningful: if a test still fails type analysis after using these fixtures,
the remaining issue is usually the object under test rather than the fixture.
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from math import isclose, sqrt
from typing import Any, ClassVar, cast

import numpy as np

from stark.core.contracts import BlockLike
from stark.core.contracts import IntervalLike
from stark.core.contracts import LinearCombine
from stark.core.contracts import Operator
from stark.methods.schemes.specialization.linear_fixed import (
    SchemeLinearFixedKernelApply,
    SchemeLinearFixedKernelDelta,
)
from stark.methods.schemes.specialization.stencil import SchemeStencil
from stark.problem import Dynamics, DynamicsStyle


@dataclass(slots=True)
class DummyScalarState:
    """One-field mutable state for tests that need scalar ODE behaviour."""

    value: float = 0.0


@dataclass(slots=True)
class DummyDynamicsInterval:
    """Minimal interval object for dynamics and signature adapter tests."""

    present: float


@dataclass(slots=True)
class DummyDynamicsState:
    """State exposing a `y` field for dynamics adapter tests."""

    y: Any


@dataclass(slots=True)
class DummyDynamicsTranslation:
    """Translation exposing a `dy` field for dynamics adapter tests."""

    dy: Any = 0.0


@dataclass(slots=True)
class DummyScalarTranslation:
    """Linear update for `DummyScalarState`.

    The object implements the full `Translation` protocol even when a test only
    needs one operation. That avoids local test doubles drifting away from the
    contract and producing type noise unrelated to the behaviour under test.
    """

    value: float = 0.0

    def __call__(
        self,
        origin: DummyScalarState,
        result: DummyScalarState,
    ) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: DummyScalarTranslation) -> DummyScalarTranslation:
        return DummyScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> DummyScalarTranslation:
        return DummyScalarTranslation(scalar * self.value)

    @staticmethod
    def scale(
        a: float,
        x: DummyScalarTranslation,
        out: DummyScalarTranslation,
    ) -> DummyScalarTranslation:
        """Write a scaled scalar translation into `out`."""

        out.value = a * x.value
        return out

    @staticmethod
    def combine2(
        scalar_left: float,
        translation_left: DummyScalarTranslation,
        scalar_right: float,
        translation_right: DummyScalarTranslation,
        out: DummyScalarTranslation,
    ) -> DummyScalarTranslation:
        """Write a two-term scalar linear combination into `out`."""

        out.value = (
            scalar_left * translation_left.value
            + scalar_right * translation_right.value
        )
        return out

    @staticmethod
    def combine3(
        scalar_a: float,
        translation_a: DummyScalarTranslation,
        scalar_b: float,
        translation_b: DummyScalarTranslation,
        scalar_c: float,
        translation_c: DummyScalarTranslation,
        out: DummyScalarTranslation,
    ) -> DummyScalarTranslation:
        """Write a three-term scalar linear combination into `out`."""

        out.value = (
            scalar_a * translation_a.value
            + scalar_b * translation_b.value
            + scalar_c * translation_c.value
        )
        return out

    linear_combine: ClassVar[LinearCombine] = (scale, combine2, combine3)


def scalar_value_combine_many(*terms: object) -> Any:
    """Write an arbitrary scalar `.value` linear combination into the output."""

    out = cast(Any, terms[-1])
    if not hasattr(out, "value"):
        raise TypeError("scalar-value combine needs an output with a value field.")

    total = 0.0
    for index in range(0, len(terms) - 1, 2):
        scalar = cast(float, terms[index])
        translation = cast(Any, terms[index + 1])
        if not hasattr(translation, "value"):
            raise TypeError("scalar-value combine terms need value fields.")
        total += scalar * translation.value
    out.value = total
    return out


scalar_value_linear_combine: LinearCombine = tuple(
    scalar_value_combine_many for _ in range(12)
)


DummyScalarTranslation.linear_combine = scalar_value_linear_combine


class DummyScalarAllocator:
    """Allocator pairing `DummyScalarState` with `DummyScalarTranslation`."""

    linear_combine = DummyScalarTranslation.linear_combine

    def allocate_state(self) -> DummyScalarState:
        return DummyScalarState()

    def copy_state(
        self,
        source: DummyScalarState,
        out: DummyScalarState,
    ) -> None:
        out.value = source.value

    def allocate_translation(self) -> DummyScalarTranslation:
        return DummyScalarTranslation()


def dummy_scalar_dynamics(rate: float) -> Dynamics:
    """Return `dy = rate * y` through STARK's public dynamics wrapper.

    Scheme and resolvent tests should use this helper when the dynamics
    itself is not under test. It keeps the fixture on the same decorated path a
    user would write while avoiding a fresh local callable class in every test
    file.
    """

    def write(
        interval: IntervalLike,
        state: DummyScalarState,
        out: DummyScalarTranslation,
    ) -> None:
        del interval
        out.value = rate * state.value

    return Dynamics(DynamicsStyle.accepts_interval_writes(write))


def dummy_quadratic_dynamics() -> Dynamics:
    """Return `dy = -y**2` through STARK's public dynamics wrapper."""

    def write(
        interval: IntervalLike,
        state: DummyScalarState,
        out: DummyScalarTranslation,
    ) -> None:
        del interval
        out.value = -(state.value ** 2)

    return Dynamics(DynamicsStyle.accepts_interval_writes(write))


def dummy_constant_dynamics(value: float = 1.0) -> Dynamics:
    """Return a constant scalar dynamics through the public wrapper."""

    def write(
        interval: IntervalLike,
        state: DummyScalarState,
        out: DummyScalarTranslation,
    ) -> None:
        del interval, state
        out.value = value

    return Dynamics(DynamicsStyle.accepts_interval_writes(write))


class DummyScalarLinearizer:
    """Configure the scalar Jacobian action `d(dy)/dy = rate`.

    The resolvent runtime supplies a mutable operator probe whose `apply`
    callable is set by the linearizer. This fixture documents that handshake
    directly so tests can focus on Newton-style scheme behaviour instead of
    rebuilding linearizer plumbing.
    """

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def __call__(
        self,
        interval: IntervalLike,
        state: DummyScalarState,
        out: Operator[DummyScalarTranslation],
    ) -> None:
        del interval
        del state

        def apply(
            translation: DummyScalarTranslation,
            result: DummyScalarTranslation,
        ) -> None:
            result.value = self.rate * translation.value

        setattr(out, "apply", apply)


@dataclass(slots=True)
class DummyArrayState:
    """NumPy-backed state for tests that need array-valued ODE behaviour."""

    y: np.ndarray

    def copy(self) -> DummyArrayState:
        """Return a deep copy suitable for independent trajectory checks."""

        return DummyArrayState(self.y.copy())


@dataclass(slots=True)
class DummyArrayTranslation:
    """NumPy-backed translation matching `DummyArrayState`.

    This fixture exercises the same array semantics that generated kernels use:
    translations are added to state in place, norms are dense Euclidean norms,
    and arithmetic returns fresh translation objects.
    """

    dy: np.ndarray
    linear_combine: ClassVar[LinearCombine]

    def __call__(self, origin: DummyArrayState, result: DummyArrayState) -> None:
        result.y[...] = origin.y + self.dy

    def norm(self) -> float:
        return float(np.linalg.norm(self.dy))

    def __add__(self, other: DummyArrayTranslation) -> DummyArrayTranslation:
        return DummyArrayTranslation(self.dy + other.dy)

    def __rmul__(self, scalar: float) -> DummyArrayTranslation:
        return DummyArrayTranslation(scalar * self.dy)


def dummy_array_combine_many(*terms: object) -> DummyArrayTranslation:
    """Write an arbitrary array linear combination into the output term."""

    out = terms[-1]
    if not isinstance(out, DummyArrayTranslation):
        raise TypeError("dummy array combine needs a DummyArrayTranslation output.")

    out.dy[...] = 0.0
    for index in range(0, len(terms) - 1, 2):
        scalar = cast(float, terms[index])
        translation = terms[index + 1]
        if not isinstance(translation, DummyArrayTranslation):
            raise TypeError("dummy array combine terms must be array translations.")
        out.dy[...] += scalar * translation.dy
    return out


DummyArrayTranslation.linear_combine = tuple(dummy_array_combine_many for _ in range(12))


class DummyArrayAllocator:
    """Allocator for fixed-size NumPy-backed dummy states."""

    linear_combine = DummyArrayTranslation.linear_combine

    def __init__(self, size: int) -> None:
        self.size = size

    def allocate_state(self) -> DummyArrayState:
        return DummyArrayState(np.zeros(self.size))

    def copy_state(
        self,
        source: DummyArrayState,
        out: DummyArrayState,
    ) -> None:
        out.y[...] = source.y

    def allocate_translation(self) -> DummyArrayTranslation:
        return DummyArrayTranslation(np.zeros(self.size))


class DummyArrayDynamics:
    """Small coupled dynamics over a three-entry array state.

    The dynamics is deliberately non-diagonal and time-dependent, so tests
    using it catch mistakes in stage state wiring rather than only scalar
    arithmetic mistakes.
    """

    def __call__(
        self,
        interval: IntervalLike,
        state: DummyArrayState,
        out: DummyArrayTranslation,
    ) -> None:
        out.dy[...] = np.array(
            [
                state.y[1] + 0.25 * interval.present,
                -state.y[0] + 0.1 * state.y[2],
                -0.5 * state.y[2] + 0.2,
            ]
        )


class DummyArrayLinearFixed:
    """Tableau fixed-linear provider for NumPy-backed dummy translations.

    Generated kernels and hand-written fixed-linear tests share the
    same in-place calling convention. This class keeps the test version in one
    place so individual tests do not reinvent subtly different fixed-linear providers.
    """

    def __call__(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        if stencil.apply:
            return self.provide_apply(stencil)
        return self.provide_delta(stencil)

    def provide_delta(
        self,
        stencil: SchemeStencil,
    ) -> SchemeLinearFixedKernelDelta[DummyArrayTranslation]:
        coefficients = tuple(stencil.coefficients)
        stencil_scale = stencil.scale

        def delta_kernel(
            step: float,
            *terms: DummyArrayTranslation,
        ) -> DummyArrayTranslation:
            *translations, out = terms
            delta = dummy_array_combine_delta(
                step,
                stencil_scale,
                coefficients,
                tuple(translations),
            )
            out.dy[...] = delta.dy
            return out

        return delta_kernel

    def provide_apply(
        self,
        stencil: SchemeStencil,
    ) -> SchemeLinearFixedKernelApply[DummyArrayState, DummyArrayTranslation]:
        """Return the apply-form kernel for a tableau stencil."""

        coefficients = tuple(stencil.coefficients)
        stencil_scale = stencil.scale

        def apply_kernel(
            step: float,
            origin: DummyArrayState,
            *terms: DummyArrayTranslation | DummyArrayState,
        ) -> DummyArrayState:
            *translations, result = terms
            assert isinstance(result, DummyArrayState)
            delta = dummy_array_combine_delta(
                step,
                stencil_scale,
                coefficients,
                tuple(translations),
            )
            delta(origin, result)
            return result

        return apply_kernel


def dummy_array_combine_delta(
    step: float,
    stencil_scale: float,
    coefficients: tuple[float, ...],
    translations: tuple[Any, ...],
) -> DummyArrayTranslation:
    """Evaluate a fixed-coefficient stencil for array translations."""

    if len(coefficients) != len(translations):
        raise AssertionError(
            f"stencil arity {len(coefficients)} received "
            f"{len(translations)} translation(s)"
        )

    if not translations:
        return DummyArrayTranslation(np.zeros(3))

    total = 0.0 * translations[0]
    for coefficient, translation in zip(coefficients, translations, strict=True):
        total = total + (step * stencil_scale * coefficient) * translation
    return total


def dummy_array_state(
    values: Sequence[float] = (1.0, -0.5, 0.25),
) -> DummyArrayState:
    """Build the standard three-entry array state used by scheme tests."""

    return DummyArrayState(np.array(values, dtype=float))


@dataclass(slots=True)
class DummyBlockScaleOperator:
    """One-block linear operator that scales a scalar translation.

    It is useful in inverter and resolvent tests that need a concrete
    block-level linear action without spelling out another local operator
    class. The operator follows the public block contract: it reads a source
    block and writes the image into a target block.
    """

    scale: float

    def __call__(
        self,
        source: BlockLike[DummyScalarTranslation],
        target: BlockLike[DummyScalarTranslation],
    ) -> BlockLike[DummyScalarTranslation]:
        target[0].value = self.scale * source[0].value
        return target


@dataclass(slots=True)
class DummyScalarEntryOperator:
    """Single-entry linear operator for scalar block-diagonal tests.

    The operator writes `scale * source` into the target translation. It also
    exposes `inverse(...)` because Jacobi-style relaxation and diagonal
    preconditioner tests need a diagonal entry with an explicit inverse action.
    """

    scale: float

    def __call__(
        self,
        source: DummyScalarTranslation,
        target: DummyScalarTranslation,
    ) -> None:
        target.value = self.scale * source.value

    def inverse(
        self,
        source: DummyScalarTranslation,
        target: DummyScalarTranslation,
    ) -> None:
        target.value = source.value / self.scale


def dummy_scalar_inner_product(
    left: DummyScalarTranslation,
    right: DummyScalarTranslation,
) -> float:
    """Return the Euclidean inner product for scalar dummy translations."""

    return left.value * right.value


@dataclass
class DummyVectorState:
    """List-backed vector state for tests that need several coordinates."""

    values: list[float]

    def __init__(self, *values: float) -> None:
        self.values = [float(value) for value in values]


@dataclass
class DummyVectorTranslation:
    """Small dense vector translation for tests that need coordinate bases.

    The scalar fixture is enough for most tests. Dense inverter and
    materialisation tests need a translation with several coordinates, so this
    object supplies the same translation operations over a Python list-backed
    vector without pulling in NumPy or engine machinery.
    """

    values: list[float]

    def __init__(self, *values: float) -> None:
        self.values = [float(value) for value in values]

    def __call__(
        self,
        origin: DummyVectorState | DummyVectorTranslation,
        result: DummyVectorState | DummyVectorTranslation,
    ) -> None:
        """Apply this vector shift to either a state or another translation.

        Most ODE paths call translations with state objects. Block and dense
        materialisation paths also use translation-shaped source and target
        buffers when probing linear operators, so this shared fixture supports
        both forms explicitly.
        """

        result.values[:] = [
            origin_value + translation_value
            for origin_value, translation_value in zip(
                origin.values,
                self.values,
                strict=True,
            )
        ]

    def norm(self) -> float:
        return sqrt(sum(value * value for value in self.values))

    def __add__(self, other: DummyVectorTranslation) -> DummyVectorTranslation:
        return DummyVectorTranslation(
            *(
                left + right
                for left, right in zip(self.values, other.values, strict=True)
            )
        )

    def __rmul__(self, scalar: float) -> DummyVectorTranslation:
        return DummyVectorTranslation(*(scalar * value for value in self.values))

    @staticmethod
    def scale(
        a: float,
        x: DummyVectorTranslation,
        out: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        """Write a scaled vector translation into `out`."""

        out.values[:] = [a * value for value in x.values]
        return out

    @staticmethod
    def combine2(
        scalar_left: float,
        translation_left: DummyVectorTranslation,
        scalar_right: float,
        translation_right: DummyVectorTranslation,
        output: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        """Write a two-term vector linear combination into `output`."""

        output.values[:] = [
            scalar_left * left_value + scalar_right * right_value
            for left_value, right_value in zip(
                translation_left.values,
                translation_right.values,
                strict=True,
            )
        ]
        return output

    @staticmethod
    def combine3(
        scalar_a: float,
        translation_a: DummyVectorTranslation,
        scalar_b: float,
        translation_b: DummyVectorTranslation,
        scalar_c: float,
        translation_c: DummyVectorTranslation,
        output: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        """Write a three-term vector linear combination into `output`."""

        output.values[:] = [
            scalar_a * value_a + scalar_b * value_b + scalar_c * value_c
            for value_a, value_b, value_c in zip(
                translation_a.values,
                translation_b.values,
                translation_c.values,
                strict=True,
            )
        ]
        return output

    linear_combine: ClassVar[LinearCombine] = (scale, combine2, combine3)


def dummy_vector_combine_many(*terms: object) -> DummyVectorTranslation:
    """Write an arbitrary vector linear combination into the output term."""

    out = terms[-1]
    if not isinstance(out, DummyVectorTranslation):
        raise TypeError("dummy vector combine needs a DummyVectorTranslation output.")

    out.values[:] = [0.0 for _ in out.values]
    for index in range(0, len(terms) - 1, 2):
        scalar = cast(float, terms[index])
        translation = terms[index + 1]
        if not isinstance(translation, DummyVectorTranslation):
            raise TypeError("dummy vector combine terms must be vector translations.")
        out.values[:] = [
            current + scalar * value
            for current, value in zip(out.values, translation.values, strict=True)
        ]
    return out


DummyVectorTranslation.linear_combine = (
    DummyVectorTranslation.scale,
    DummyVectorTranslation.combine2,
    DummyVectorTranslation.combine3,
    *(dummy_vector_combine_many for _ in range(9)),
)


class DummyVectorBasis:
    """Coordinate basis for `DummyVectorTranslation`.

    Dense inverter tests use this to exercise the real materialisation path
    with a transparent list-backed vector. The basis follows the public
    `TranslationBasisLike` contract, including accepting generic mutable coordinate
    buffers rather than only concrete lists.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def vector(
        self,
        index: int,
        output: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        output.values[:] = [0.0] * self.dimension
        output.values[index] = 1.0
        return output

    def coordinate(self, index: int, translation: DummyVectorTranslation) -> float:
        return translation.values[index]

    def coordinates(
        self,
        translation: DummyVectorTranslation,
        output: MutableSequence[float],
    ) -> MutableSequence[float]:
        for index, value in enumerate(translation.values):
            output[index] = value
        return output

    def synthesize(
        self,
        coordinates: Sequence[float],
        output: DummyVectorTranslation,
    ) -> DummyVectorTranslation:
        output.values[:] = [float(value) for value in coordinates]
        return output


class DummyVectorAllocator:
    """Allocator for `DummyVectorState` and `DummyVectorTranslation`."""

    linear_combine = DummyVectorTranslation.linear_combine

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.basis = DummyVectorBasis(dimension)

    def allocate_state(self) -> DummyVectorState:
        return DummyVectorState(*([0.0] * self.dimension))

    def copy_state(
        self,
        source: DummyVectorState,
        out: DummyVectorState,
    ) -> None:
        out.values[:] = source.values[:]

    def allocate_translation(self) -> DummyVectorTranslation:
        return DummyVectorTranslation(*([0.0] * self.dimension))


def assert_dummy_vector_close(
    actual: DummyVectorTranslation,
    expected: Sequence[float],
) -> None:
    """Assert exact-length vector agreement with a tight numeric tolerance."""

    assert len(actual.values) == len(expected)
    for actual_value, expected_value in zip(actual.values, expected, strict=True):
        assert isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12)


def dummy_zero_rhs(
    interval: IntervalLike,
    state: DummyScalarState,
    out: DummyScalarTranslation,
) -> None:
    """Write the zero dynamics into `out`."""

    del interval, state
    out.value = 0.0


def dummy_exponential_growth_rhs(
    interval: IntervalLike,
    state: DummyScalarState,
    out: DummyScalarTranslation,
) -> None:
    """Write `y' = y` for the scalar state into `out`."""

    del interval
    out.value = state.value


class DummyTableauLinearFixed:
    """Small fixed-linear provider that evaluates tableau stencils directly.

    It lets tests exercise specialized scheme paths without depending on a
    backend code generator. The kernels it returns obey the same in-place
    calling convention as generated fixed-linear kernels.
    """

    def __call__(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        if stencil.apply:
            return self.provide_apply(stencil)
        return self.provide_delta(stencil)

    def provide_delta(
        self,
        stencil: SchemeStencil,
    ) -> SchemeLinearFixedKernelDelta[DummyScalarTranslation]:
        coefficients = tuple(stencil.coefficients)
        fixed_scale = stencil.scale

        def delta_kernel(
            step: float,
            *terms: DummyScalarTranslation,
        ) -> DummyScalarTranslation:
            *translations, out = terms
            delta = self.combine_delta(
                step,
                fixed_scale,
                coefficients,
                tuple(translations),
            )
            out.value = delta.value
            return out

        return delta_kernel

    def provide_apply(
        self,
        stencil: SchemeStencil,
    ) -> SchemeLinearFixedKernelApply[DummyScalarState, DummyScalarTranslation]:
        """Return the apply-form kernel for a tableau stencil."""

        coefficients = tuple(stencil.coefficients)
        fixed_scale = stencil.scale

        def apply_kernel(
            step: float,
            origin: DummyScalarState,
            *terms: DummyScalarTranslation | DummyScalarState,
        ) -> DummyScalarState:
            *translations, result = terms
            if not isinstance(result, DummyScalarState):
                raise TypeError("DummyTableauLinearFixed apply kernel needs an output buffer.")
            delta = self.combine_delta(
                step,
                fixed_scale,
                coefficients,
                tuple(translations),
            )
            delta(origin, result)
            return result

        return apply_kernel

    @staticmethod
    def combine_delta(
        step: float,
        stencil_scale: float,
        coefficients: tuple[float, ...],
        translations: tuple[Any, ...],
    ) -> DummyScalarTranslation:
        """Evaluate a fixed-coefficient stencil as a scalar translation."""

        if len(coefficients) != len(translations):
            raise AssertionError(
                f"stencil arity {len(coefficients)} received "
                f"{len(translations)} translation(s)"
            )

        if not translations:
            return DummyScalarTranslation()

        total = 0.0 * translations[0]
        for coefficient, translation in zip(coefficients, translations, strict=True):
            total = total + (step * stencil_scale * coefficient) * translation
        return total


class DummyValueLinearFixed:
    """Fixed-linear provider for tests with `.value` translations.

    This is deliberately a small duck-typed test helper rather than an engine
    generator. It lets scheme tests exercise their specialized call paths
    without depending on an engine generator.
    """

    def __call__(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        if stencil.apply:
            return self.provide_apply(stencil)
        return self.provide_delta(stencil)

    def provide_delta(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        coefficients = tuple(stencil.coefficients)
        fixed_scale = stencil.scale

        def delta_kernel(
            step: float,
            *terms: Any,
        ) -> Any:
            *translations, out = terms
            if not translations:
                self.write_value(out, 0.0)
                return out

            delta = self.combine_delta(
                step,
                fixed_scale,
                coefficients,
                tuple(translations),
            )
            out.value = delta.value
            return out

        return delta_kernel

    def provide_apply(
        self,
        stencil: SchemeStencil,
    ) -> Any:
        coefficients = tuple(stencil.coefficients)
        fixed_scale = stencil.scale

        def apply_kernel(
            step: float,
            origin: Any,
            *terms: Any,
        ) -> Any:
            *translations, result = terms
            if not translations:
                self.write_value(result, origin.value)
                return result

            delta = self.combine_delta(
                step,
                fixed_scale,
                coefficients,
                tuple(translations),
            )
            delta(origin, result)
            return result

        return apply_kernel

    @staticmethod
    def combine_delta(
        step: float,
        stencil_scale: float,
        coefficients: tuple[float, ...],
        translations: tuple[Any, ...],
    ) -> Any:
        if len(coefficients) != len(translations):
            raise AssertionError(
                f"stencil arity {len(coefficients)} received "
                f"{len(translations)} translation(s)"
            )

        if not translations:
            raise AssertionError("value fixed-linear tests need at least one term.")

        total = 0.0 * translations[0]
        for coefficient, translation in zip(coefficients, translations, strict=True):
            total = total + (step * stencil_scale * coefficient) * translation
        return total

    @staticmethod
    def write_value(target: Any, value: Any) -> None:
        try:
            target.value[...] = value
        except TypeError:
            target.value = value


__all__ = [
    "DummyArrayAllocator",
    "DummyArrayDynamics",
    "DummyArrayLinearFixed",
    "DummyArrayState",
    "DummyArrayTranslation",
    "DummyScalarAllocator",
    "DummyScalarLinearizer",
    "DummyBlockScaleOperator",
    "DummyDynamicsInterval",
    "DummyDynamicsState",
    "DummyDynamicsTranslation",
    "DummyScalarEntryOperator",
    "DummyScalarState",
    "DummyScalarTranslation",
    "DummyTableauLinearFixed",
    "DummyValueLinearFixed",
    "DummyVectorAllocator",
    "DummyVectorBasis",
    "DummyVectorState",
    "DummyVectorTranslation",
    "assert_dummy_vector_close",
    "dummy_array_combine_delta",
    "dummy_array_state",
    "dummy_constant_dynamics",
    "dummy_quadratic_dynamics",
    "dummy_scalar_dynamics",
    "dummy_scalar_inner_product",
    "dummy_exponential_growth_rhs",
    "dummy_zero_rhs",
    "scalar_value_linear_combine",
]
