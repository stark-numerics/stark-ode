from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from stark import Executor, Interval, ExecutorTolerance
from stark.accelerators import Accelerator
from stark.algebraist.runtime import AlgebraistRuntimeSpecialist
from stark.resolvents import ResolventPicard
from stark.resolvents.support.policy import ResolventPolicy
from stark.schemes.imex_adaptive.kennedy_carpenter32 import (
    SchemeKennedyCarpenter32,
)
from stark.schemes.imex_adaptive.kennedy_carpenter43_6 import (
    SchemeKennedyCarpenter43_6,
)
from stark.schemes.imex_adaptive.kennedy_carpenter43_7 import (
    SchemeKennedyCarpenter43_7,
)
from stark.schemes.imex_adaptive.kennedy_carpenter54 import (
    SchemeKennedyCarpenter54,
)
from stark.schemes.imex_adaptive.kennedy_carpenter54b import (
    SchemeKennedyCarpenter54b,
)


IMEX_ADAPTIVE_SCHEMES = (
    SchemeKennedyCarpenter32,
    SchemeKennedyCarpenter43_6,
    SchemeKennedyCarpenter43_7,
    SchemeKennedyCarpenter54,
    SchemeKennedyCarpenter54b,
)


@dataclass(slots=True)
class ArrayScalarState:
    value: np.ndarray

    @classmethod
    def zero(cls) -> "ArrayScalarState":
        return cls(np.zeros(1))


@dataclass(slots=True)
class ArrayScalarTranslation:
    value: np.ndarray

    @classmethod
    def zero(cls) -> "ArrayScalarTranslation":
        return cls(np.zeros(1))

    def __call__(self, origin: ArrayScalarState, result: ArrayScalarState) -> None:
        result.value[...] = origin.value + self.value

    def norm(self) -> float:
        return float(abs(self.value[0]))

    def __add__(self, other: "ArrayScalarTranslation") -> "ArrayScalarTranslation":
        return ArrayScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ArrayScalarTranslation":
        return ArrayScalarTranslation(scalar * self.value)


class ArrayScalarAllocator:
    def allocate_state(self) -> ArrayScalarState:
        return ArrayScalarState.zero()

    def copy_state(self, source: ArrayScalarState, out: ArrayScalarState) -> None:
        out.value[...] = source.value

    def allocate_translation(self) -> ArrayScalarTranslation:
        return ArrayScalarTranslation.zero()


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def array_explicit_rhs(
    interval: Interval,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 1.0


def array_implicit_rhs(
    interval: Interval,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 2.0


def make_array_scheme(
    scheme_cls: type[Any],
    *,
    specialist: bool = False,
) -> Any:
    allocator = ArrayScalarAllocator()
    derivative = SplitDerivative(
        explicit=array_explicit_rhs,
        implicit=array_implicit_rhs,
    )
    resolvent = ResolventPicard(
        allocator,
        ExecutorTolerance=ExecutorTolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        derivative,
        allocator,
        resolvent=resolvent,
        specialist=(
            AlgebraistRuntimeSpecialist(
                translation=allocator.allocate_translation(),
                allocator=allocator,
            )
            if specialist
            else None
        ),
    )


def tight_executor() -> Executor:
    return Executor(tolerance=ExecutorTolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_specialist_path_is_scheme_owned_generated_call(
    scheme_cls: type[Any],
) -> None:
    scheme = make_array_scheme(scheme_cls, specialist=True)

    assert hasattr(scheme_cls, "call_specialized")
    assert scheme.call_monitorable.__self__ is scheme
    assert scheme.call_monitorable.__func__ is scheme_cls.call_specialized
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_specialist_path_matches_generic_path(
    scheme_cls: type[Any],
) -> None:
    generic = make_array_scheme(scheme_cls)
    generated = make_array_scheme(scheme_cls, specialist=True)
    generic_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generated_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state, tight_executor())
    generated_dt = generated(generated_interval, generated_state, tight_executor())

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_specialist_path_prepares_expected_kernel_family(
    scheme_cls: type[Any],
) -> None:
    scheme = make_array_scheme(scheme_cls, specialist=True)

    stage_count = len(scheme_cls.tableau.c)
    assert len(scheme.stage_rhs_kernels) == stage_count
    assert callable(scheme.advance_delta_kernel)
    assert callable(scheme.error_delta_kernel)
