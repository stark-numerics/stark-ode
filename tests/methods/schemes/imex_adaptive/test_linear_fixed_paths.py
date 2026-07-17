from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from stark import Interval, Tolerance
from stark.engines.accelerators import AcceleratorNone
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.imex.adaptive.kennedy_carpenter32 import (
    SchemeKennedyCarpenter32,
)
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_6 import (
    SchemeKennedyCarpenter43_6,
)
from stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7 import (
    SchemeKennedyCarpenter43_7,
)
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54 import (
    SchemeKennedyCarpenter54,
)
from stark.methods.schemes.imex.adaptive.kennedy_carpenter54b import (
    SchemeKennedyCarpenter54b,
)
from tests.support import DummyValueLinearFixed, scalar_value_linear_combine


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
    linear_combine = scalar_value_linear_combine

    def allocate_state(self) -> ArrayScalarState:
        return ArrayScalarState.zero()

    def copy_state(self, source: ArrayScalarState, out: ArrayScalarState) -> None:
        out.value[...] = source.value

    def allocate_translation(self) -> ArrayScalarTranslation:
        return ArrayScalarTranslation.zero()


@dataclass(slots=True)
class SplitDynamics:
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
    linear_fixed: bool = False,
) -> Any:
    allocator = ArrayScalarAllocator()
    dynamics = SplitDynamics(
        explicit=array_explicit_rhs,
        implicit=array_implicit_rhs,
    )
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        dynamics,
        allocator,
        resolvent=resolvent,
        linear_fixed=DummyValueLinearFixed() if linear_fixed else None,
    )


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_linear_fixed_path_is_scheme_owned_generated_call(
    scheme_cls: type[Any],
) -> None:
    scheme = make_array_scheme(scheme_cls, linear_fixed=True)

    assert hasattr(scheme_cls, "call_specialized")


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_linear_fixed_path_matches_generic_path(
    scheme_cls: type[Any],
) -> None:
    generic = make_array_scheme(scheme_cls)
    generated = make_array_scheme(scheme_cls, linear_fixed=True)
    generic_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generated_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state)
    generated_dt = generated(generated_interval, generated_state)

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
def test_imex_adaptive_linear_fixed_path_prepares_expected_kernel_family(
    scheme_cls: type[Any],
) -> None:
    scheme = make_array_scheme(scheme_cls, linear_fixed=True)
    adaptive_step = scheme.adaptive_step

    stage_count = len(scheme_cls.tableau.c)
    assert len(adaptive_step.stage_rhs_kernels) == stage_count
    assert callable(adaptive_step.advance_delta_kernel)
    assert callable(adaptive_step.error_delta_kernel)
