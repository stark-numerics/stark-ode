from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist.runtime import AlgebraistRuntimeSpecialist
from stark.resolvents import ResolventCoupledPicard
from stark.resolvents.support.policy import ResolventPolicy
from stark.schemes.implicit_fixed.gauss_legendre4 import SchemeGaussLegendre4
from stark.schemes.implicit_fixed.lobatto_iiic4 import SchemeLobattoIIIC4
from stark.schemes.implicit_fixed.radau_iia5 import SchemeRadauIIA5


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: ScalarTranslation) -> ScalarTranslation:
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> ScalarTranslation:
        return ScalarTranslation(scalar * self.value)


class ScalarWorkbench:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, dst: ScalarState, src: ScalarState) -> None:
        dst.value = src.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


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


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def array_constant_rhs(
    interval: Interval,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 1.0


def make_scheme(scheme_cls):
    workbench = ScalarWorkbench()
    resolvent = ResolventCoupledPicard(
        workbench,
        tableau=scheme_cls.tableau,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
    )
    return scheme_cls(
        constant_rhs,
        workbench,
        resolvent=resolvent,
    )


def make_array_scheme(scheme_cls, *, specialist: bool = False):
    workbench = ArrayScalarWorkbench()
    resolvent = ResolventCoupledPicard(
        workbench,
        tableau=scheme_cls.tableau,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
    )
    return scheme_cls(
        array_constant_rhs,
        workbench,
        resolvent=resolvent,
        specialist=(
            AlgebraistRuntimeSpecialist(
                translation=workbench.allocate_translation(),
                workbench=workbench,
            )
            if specialist
            else None
        ),
    )


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_default_call_path_is_scheme_owned_generic_call(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_inline
    assert scheme.redirect_call == scheme.call_pure


def test_gauss_legendre4_specialist_path_is_scheme_owned_generated_call() -> None:
    scheme = make_array_scheme(SchemeGaussLegendre4, specialist=True)

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeGaussLegendre4.call_specialized
    assert scheme.redirect_call == scheme.call_pure


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_stiffly_accurate_collocation_accepts_no_op_specialist_path(
    scheme_cls,
) -> None:
    scheme = make_array_scheme(scheme_cls, specialist=True)

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_specialized
    assert scheme.redirect_call == scheme.call_pure


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
        replacement_executor: Executor,
    ) -> float:
        del replacement_interval, replacement_executor
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_call_performs_one_constant_rhs_step(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(0.125)
    assert interval.step == pytest.approx(0.125)


def test_gauss_legendre4_specialist_path_matches_generic_path() -> None:
    generic = make_array_scheme(SchemeGaussLegendre4)
    generated = make_array_scheme(SchemeGaussLegendre4, specialist=True)
    generic_interval = Interval(present=0.0, step=0.125, stop=1.0)
    generated_interval = Interval(present=0.0, step=0.125, stop=1.0)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state, Executor())
    generated_dt = generated(generated_interval, generated_state, Executor())

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


def test_gauss_legendre4_specialist_kernel_is_prepared() -> None:
    scheme = make_array_scheme(SchemeGaussLegendre4, specialist=True)

    assert callable(scheme.advance_update)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_call_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(0.05)
    assert interval.step == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_returns_zero_when_interval_is_complete(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=1.0, step=0.125, stop=1.0)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.0)
    assert state.value == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeGaussLegendre4,
        SchemeLobattoIIIC4,
        SchemeRadauIIA5,
    ],
)
def test_collocation_fixed_snapshot_and_safety_are_exposed_through_scheme(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)

    scheme.set_apply_delta_safety(False)
    scheme.set_apply_delta_safety(True)
