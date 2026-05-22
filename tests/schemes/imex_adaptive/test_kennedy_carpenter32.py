from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Executor, Interval, Tolerance
from stark.accelerators import Accelerator
from stark.algebraist.classic import (
    Algebraist,
    AlgebraistBroadcast,
    AlgebraistField,
    AlgebraistLooped,
    AlgebraistSmallFixed,
)
from stark.monitor import Monitor
from stark.resolvents import ResolventPicard
from stark.resolvents.support.policy import ResolventPolicy
from stark.schemes.imex_adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32


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


@dataclass(slots=True)
class SplitDerivative:
    explicit: object
    implicit: object


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


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


def make_scheme() -> SchemeKennedyCarpenter32:
    workbench = ScalarWorkbench()
    derivative = SplitDerivative(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        zero_rhs,
        workbench,
        tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12),
        policy=ResolventPolicy(max_iterations=8),
        accelerator=Accelerator.none(),
        tableau=SchemeKennedyCarpenter32.tableau,
    )
    return SchemeKennedyCarpenter32(
        derivative,
        workbench,
        resolvent=resolvent,
    )


def make_array_scheme(
    *,
    algebraist: Algebraist | None = None,
) -> SchemeKennedyCarpenter32:
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
        tableau=SchemeKennedyCarpenter32.tableau,
    )
    return SchemeKennedyCarpenter32(
        derivative,
        workbench,
        resolvent=resolvent,
        algebraist=algebraist,
    )


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


def test_kennedy_carpenter32_owns_converted_call_surface() -> None:
    assert "__call__" in SchemeKennedyCarpenter32.__dict__
    assert "call_bind" in SchemeKennedyCarpenter32.__dict__
    assert "call_generic" in SchemeKennedyCarpenter32.__dict__
    assert "call_algebraist" in SchemeKennedyCarpenter32.__dict__
    assert "call_monitored" in SchemeKennedyCarpenter32.__dict__

    scheme = make_scheme()

    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__
    assert scheme.call_pure.__func__ is scheme.call_generic.__func__


@pytest.mark.parametrize(
    "field",
    [
        AlgebraistField("value", "value", policy=AlgebraistBroadcast()),
        AlgebraistField("value", "value", policy=AlgebraistLooped(rank=1)),
        AlgebraistField("value", "value", policy=AlgebraistSmallFixed(shape=(1,))),
    ],
)
def test_kennedy_carpenter32_algebraist_path_is_scheme_owned_generated_call(
    field: AlgebraistField,
) -> None:
    scheme = make_array_scheme(algebraist=Algebraist(fields=(field,)))

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is SchemeKennedyCarpenter32.call_algebraist
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_bind.__func__


def test_kennedy_carpenter32_accepts_zero_split_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)
    assert scheme.redirect_call.__func__ is scheme.call_pure.__func__

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.proposed_dt == pytest.approx(0.1)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


@pytest.mark.parametrize(
    "field",
    [
        AlgebraistField("value", "value", policy=AlgebraistBroadcast()),
        AlgebraistField("value", "value", policy=AlgebraistLooped(rank=1)),
        AlgebraistField("value", "value", policy=AlgebraistSmallFixed(shape=(1,))),
    ],
)
def test_kennedy_carpenter32_algebraist_path_matches_generic_path(
    field: AlgebraistField,
) -> None:
    algebraist = Algebraist(fields=(field,))
    generic = make_array_scheme()
    generated = make_array_scheme(algebraist=algebraist)
    generic_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generated_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state, tight_executor())
    generated_dt = generated(generated_interval, generated_state, tight_executor())

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


def test_kennedy_carpenter32_algebraist_source_is_inspectable() -> None:
    algebraist = Algebraist(fields=(AlgebraistField("value", "value"),))

    make_array_scheme(algebraist=algebraist)

    assert "stage1_shift_combine" in algebraist.sources
    assert "stage2_shift_combine" in algebraist.sources
    assert "stage3_shift_combine" in algebraist.sources
    assert "high_delta_combine" in algebraist.sources
    assert "error_delta_combine" in algebraist.sources
    assert "low_delta_combine" not in algebraist.sources
    assert "explicit_k0" in algebraist.sources["stage1_shift_combine"]
    assert "implicit_k0" in algebraist.sources["stage1_shift_combine"]


def test_kennedy_carpenter32_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.25, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(2.0)

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.05)
    assert report.proposed_dt == pytest.approx(0.05)


def test_kennedy_carpenter32_monitoring_uses_scheme_owned_boundary() -> None:
    scheme = make_scheme()
    monitor = Monitor()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    executor = Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    scheme.assign_executor(executor)
    scheme.assign_monitor(monitor.scheme)

    assert scheme.redirect_call.__func__ is scheme.call_monitored.__func__

    accepted_dt = scheme(interval, state, executor)

    assert accepted_dt == pytest.approx(0.1)
    assert len(monitor.scheme.adaptive_steps) == 1

    record = monitor.scheme.adaptive_steps[0]
    assert record.scheme == scheme.short_name
    assert record.t_start == pytest.approx(0.0)
    assert record.t_end == pytest.approx(0.1)
    assert record.proposed_dt == pytest.approx(0.1)
    assert record.accepted_dt == pytest.approx(0.1)
    assert record.error_ratio == pytest.approx(0.0)
    assert record.rejection_count == 0
