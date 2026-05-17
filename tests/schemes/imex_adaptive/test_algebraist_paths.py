from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist import (
    Algebraist,
    AlgebraistBroadcast,
    AlgebraistField,
    AlgebraistLooped,
    AlgebraistSmallFixed,
)
from stark.resolvents import ResolventPicard
from stark.resolvents.policy import ResolventPolicy
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


class ArrayScalarWorkbench:
    def allocate_state(self) -> ArrayScalarState:
        return ArrayScalarState.zero()

    def copy_state(self, dst: ArrayScalarState, src: ArrayScalarState) -> None:
        dst.value[...] = src.value

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
    algebraist: Algebraist | None = None,
) -> Any:
    workbench = ArrayScalarWorkbench()
    derivative = SplitDerivative(
        explicit=array_explicit_rhs,
        implicit=array_implicit_rhs,
    )
    resolvent = ResolventPicard(
        array_implicit_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        derivative,
        workbench,
        resolvent=resolvent,
        algebraist=algebraist,
    )


def algebraist_fields() -> tuple[AlgebraistField, ...]:
    return (
        AlgebraistField("value", "value", policy=AlgebraistBroadcast()),
        AlgebraistField("value", "value", policy=AlgebraistLooped(rank=1)),
        AlgebraistField("value", "value", policy=AlgebraistSmallFixed(shape=(1,))),
    )


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
@pytest.mark.parametrize("field", algebraist_fields())
def test_imex_adaptive_algebraist_path_is_scheme_owned_generated_call(
    scheme_cls: type[Any],
    field: AlgebraistField,
) -> None:
    scheme = make_array_scheme(scheme_cls, algebraist=Algebraist(fields=(field,)))

    assert "call_algebraist" in scheme_cls.__dict__
    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_algebraist
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__


@pytest.mark.parametrize("scheme_cls", IMEX_ADAPTIVE_SCHEMES)
@pytest.mark.parametrize("field", algebraist_fields())
def test_imex_adaptive_algebraist_path_matches_generic_path(
    scheme_cls: type[Any],
    field: AlgebraistField,
) -> None:
    algebraist = Algebraist(fields=(field,))
    generic = make_array_scheme(scheme_cls)
    generated = make_array_scheme(scheme_cls, algebraist=algebraist)
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
def test_imex_adaptive_algebraist_source_is_inspectable(
    scheme_cls: type[Any],
) -> None:
    algebraist = Algebraist(fields=(AlgebraistField("value", "value"),))

    make_array_scheme(scheme_cls, algebraist=algebraist)

    stage_count = len(scheme_cls.tableau.c)
    for stage_index in range(1, stage_count):
        assert f"stage{stage_index}_shift_combine" in algebraist.sources

    assert "high_delta_combine" in algebraist.sources
    assert "error_delta_combine" in algebraist.sources
    assert "low_delta_combine" not in algebraist.sources
    assert "explicit_k0" in algebraist.sources["stage1_shift_combine"]
    assert "implicit_k0" in algebraist.sources["stage1_shift_combine"]
