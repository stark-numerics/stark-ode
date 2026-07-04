from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Interval, Tolerance
from stark.core.contracts import DynamicsLike, IntervalLike
from stark.engines.shared.accelerators import AcceleratorNone
from stark.engines.shared.algebraist.runtime import AlgebraistRuntimeSpecialist
from stark.diagnostics.monitor import Monitor
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.imex.adaptive.kennedy_carpenter32 import SchemeKennedyCarpenter32


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


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

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


class ArrayScalarAllocator:
    def allocate_state(self) -> ArrayScalarState:
        return ArrayScalarState.zero()

    def copy_state(self, source: ArrayScalarState, out: ArrayScalarState) -> None:
        out.value[...] = source.value

    def allocate_translation(self) -> ArrayScalarTranslation:
        return ArrayScalarTranslation.zero()


@dataclass(slots=True)
class SplitDynamics:
    explicit: DynamicsLike
    implicit: DynamicsLike


def zero_rhs(
    interval: IntervalLike,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def array_explicit_rhs(
    interval: IntervalLike,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 1.0


def array_implicit_rhs(
    interval: IntervalLike,
    state: ArrayScalarState,
    out: ArrayScalarTranslation,
) -> None:
    del interval, state
    out.value[...] = 2.0


def make_scheme(*, monitor=None) -> SchemeKennedyCarpenter32:
    allocator = ScalarAllocator()
    dynamics = SplitDynamics(
        explicit=zero_rhs,
        implicit=zero_rhs,
    )
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeKennedyCarpenter32.tableau,
    )
    return SchemeKennedyCarpenter32(
        dynamics,
        allocator,
        resolvent=resolvent,
        monitor=monitor,
    )


def make_array_scheme(
    *,
    specialist: bool = False,
) -> SchemeKennedyCarpenter32:
    allocator = ArrayScalarAllocator()
    dynamics = SplitDynamics(
        explicit=array_explicit_rhs,
        implicit=array_implicit_rhs,
    )
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=SchemeKennedyCarpenter32.tableau,
    )
    return SchemeKennedyCarpenter32(
        dynamics,
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


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))
def test_kennedy_carpenter32_accepts_zero_split_step() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(2.0)

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.1)
    assert report.proposed_dt == pytest.approx(0.1)
    assert report.error_ratio == pytest.approx(0.0)
    assert report.rejection_count == 0


def test_kennedy_carpenter32_specialist_path_matches_generic_path() -> None:
    generic = make_array_scheme()
    generated = make_array_scheme(specialist=True)
    generic_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generated_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state)
    generated_dt = generated(generated_interval, generated_state)

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


def test_kennedy_carpenter32_specialist_path_prepares_expected_kernel_family() -> None:
    scheme = make_array_scheme(specialist=True)
    adaptive_step = scheme.adaptive_step

    assert len(adaptive_step.stage_rhs_kernels) == len(SchemeKennedyCarpenter32.tableau.c)
    assert callable(adaptive_step.advance_delta_kernel)
    assert callable(adaptive_step.error_delta_kernel)


def test_kennedy_carpenter32_clips_to_remaining_interval() -> None:
    scheme = make_scheme()
    interval = Interval(present=0.25, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(2.0)

    report = scheme.step_control.report()
    assert report.accepted_dt == pytest.approx(0.05)
    assert report.proposed_dt == pytest.approx(0.05)


def test_kennedy_carpenter32_monitoring_uses_scheme_owned_boundary() -> None:
    monitor = Monitor()
    scheme = make_scheme(monitor=monitor.scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    configuration = Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))

    assert scheme.monitor is monitor.scheme
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert len(monitor.scheme.adaptive_steps) == 1

    record = monitor.scheme.adaptive_steps[0]
    assert record.scheme == scheme.descriptor.short_name
    assert record.t_start == pytest.approx(0.0)
    assert record.t_end == pytest.approx(0.1)
    assert record.proposed_dt == pytest.approx(0.1)
    assert record.accepted_dt == pytest.approx(0.1)
    assert record.error_ratio == pytest.approx(0.0)
    assert record.rejection_count == 0
