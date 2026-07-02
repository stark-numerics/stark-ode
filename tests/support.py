"""Shared typed test fixtures for STARK's public contracts.

These helpers are intentionally tiny but complete. They satisfy the same
state, translation, allocator, derivative, and specialist contracts that user
code is expected to satisfy, which keeps tests focused on the behaviour under
test instead of on half-real local fakes.

The scalar fixtures are useful for scheme, resolvent, and inverter tests that
only need a one-dimensional state space. They make Pyright complaints more
meaningful: if a test still fails type analysis after using these fixtures,
the remaining issue is usually the object under test rather than the fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stark.core.contracts import BlockLike
from stark.core.contracts import IntervalLike
from stark.methods.schemes.specialization.stencil import SchemeStencil


@dataclass(slots=True)
class DummyScalarState:
    """One-field mutable state for tests that need scalar ODE behaviour."""

    value: float = 0.0


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
        scalar: float,
        translation: DummyScalarTranslation,
        out: DummyScalarTranslation,
    ) -> DummyScalarTranslation:
        """Write a scaled scalar translation into `out`."""

        out.value = scalar * translation.value
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

    linear_combine = [scale, combine2, combine3]


class DummyScalarAllocator:
    """Allocator pairing `DummyScalarState` with `DummyScalarTranslation`."""

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


def dummy_zero_rhs(
    interval: IntervalLike,
    state: DummyScalarState,
    out: DummyScalarTranslation,
) -> None:
    """Write the zero derivative into `out`."""

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


class DummyTableauSpecialist:
    """Small specialist that evaluates tableau stencils directly.

    It lets tests exercise specialized scheme paths without depending on a
    backend code generator. The kernels it returns obey the same in-place
    calling convention as generated specialist kernels.
    """

    def provide_delta(self, stencil: SchemeStencil):
        coefficients = tuple(stencil.coefficients)
        fixed_scale = stencil.scale

        if stencil.apply:

            def apply_kernel(
                step: float,
                origin: DummyScalarState,
                *terms: Any,
            ) -> DummyScalarState:
                *translations, result = terms
                delta = self.combine_delta(
                    step,
                    fixed_scale,
                    coefficients,
                    tuple(translations),
                )
                delta(origin, result)
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms: Any,
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

    def provide_apply(self, stencil: SchemeStencil):
        """Return the apply-form kernel for a tableau stencil."""

        return self.provide_delta(stencil)

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


__all__ = [
    "DummyScalarAllocator",
    "DummyBlockScaleOperator",
    "DummyScalarState",
    "DummyScalarTranslation",
    "DummyTableauSpecialist",
    "dummy_exponential_growth_rhs",
    "dummy_zero_rhs",
]
