from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from stark import Integrator, Interval, IntegratorStepper, Tolerance
from stark.engines.accelerators import AcceleratorNone
from stark.engines.algebraist.runtime import AlgebraistRuntimeSpecialist
from stark.monitor import Monitor
from stark.methods.resolvents import ResolventPicard
from stark import Configuration
from stark.methods.schemes.implicit.adaptive.kvaerno3 import SchemeKvaerno3
from stark.methods.schemes.implicit.adaptive.kvaerno4 import SchemeKvaerno4
from stark.methods.schemes.implicit.adaptive.sdirk21 import SchemeSDIRK21


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


def make_scheme(scheme_cls, *, monitor=None):
    allocator = ScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        constant_rhs,
        allocator,
        resolvent=resolvent,
        monitor=monitor,
    )


def make_array_scalar_scheme(
    scheme_cls,
    *,
    specialist: bool = False,
):
    allocator = ArrayScalarAllocator()
    resolvent = ResolventPicard(
        allocator,
        configuration=Configuration(resolvent_tolerance=Tolerance(atol=1.0e-12, rtol=1.0e-12), resolvent_maximum_steps=8),
        accelerator=AcceleratorNone(),
        tableau=scheme_cls.tableau,
    )
    return scheme_cls(
        array_constant_rhs,
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


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_default_call_path_is_scheme_owned_generic_call(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is scheme_cls.call_inline

    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_specialist_path_is_scheme_owned_generated_call(
    scheme_cls,
) -> None:
    scheme = make_array_scalar_scheme(scheme_cls, specialist=True)

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is scheme_cls.call_specialized

    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
    ) -> float:
        del replacement_interval
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_call_returns_accepted_dt_and_updates_next_step(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert state.value == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_call_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = make_scheme(scheme_cls)
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(0.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.2)
    assert state.value == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_specialist_path_matches_generic_path(
    scheme_cls,
) -> None:
    generic = make_array_scalar_scheme(scheme_cls)
    generated = make_array_scalar_scheme(scheme_cls, specialist=True)
    generic_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generated_interval = Interval(present=0.0, step=0.1, stop=0.3)
    generic_state = ArrayScalarState.zero()
    generated_state = ArrayScalarState.zero()

    generic_dt = generic(generic_interval, generic_state)
    generated_dt = generated(generated_interval, generated_state)

    assert generated_dt == pytest.approx(generic_dt)
    assert generated_state.value[0] == pytest.approx(generic_state.value[0])
    assert generated_interval.step == pytest.approx(generic_interval.step)


@pytest.mark.parametrize(
    ("scheme_cls", "known_names"),
    [
        (SchemeSDIRK21, ("known2", "known3")),
        (SchemeKvaerno3, ("known2", "known3", "known4")),
        (SchemeKvaerno4, ("known2", "known3", "known4", "known5")),
    ],
)
def test_esdirk_adaptive_specialist_path_prepares_expected_kernel_family(
    scheme_cls,
    known_names: tuple[str, ...],
) -> None:
    scheme = make_array_scalar_scheme(scheme_cls, specialist=True)

    for known_name in known_names:
        assert callable(getattr(scheme, f"{known_name}_call"))

    assert callable(scheme.high_delta_call)
    assert callable(scheme.error_delta_call)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeSDIRK21,
        SchemeKvaerno3,
        SchemeKvaerno4,
    ],
)
def test_esdirk_adaptive_snapshot_is_exposed_through_scheme(
    scheme_cls,
) -> None:
    scheme = make_scheme(scheme_cls)
    state = ScalarState(3.0)

    snapshot = scheme.snapshot_state(state)

    assert snapshot is not state
    assert snapshot.value == pytest.approx(3.0)

    state.value = 9.0

    assert snapshot.value == pytest.approx(3.0)

@pytest.mark.parametrize(
    ("scheme_cls", "scheme_name"),
    [
        (SchemeSDIRK21, "SDIRK21"),
        (SchemeKvaerno3, "Kvaerno3"),
        (SchemeKvaerno4, "Kvaerno4"),
    ],
)
def test_esdirk_adaptive_monitoring_records_existing_adaptive_fields(
    scheme_cls,
    scheme_name: str,
) -> None:
    monitor = Monitor()
    scheme = make_scheme(scheme_cls, monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(0.0)

    list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(monitor.scheme.adaptive_steps) == 2

    first = monitor.scheme.adaptive_steps[0]
    second = monitor.scheme.adaptive_steps[1]

    assert first.scheme == scheme_name
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio >= 0.0
    assert first.error_ratio < 1.0e-6
    assert first.rejection_count == 0

    assert second.scheme == scheme_name
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio >= 0.0
    assert second.error_ratio < 1.0e-6
    assert second.rejection_count == 0
