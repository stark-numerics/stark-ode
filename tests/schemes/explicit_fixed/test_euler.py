from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Interval
from stark.diagnostics.monitor import Monitor
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.specialization.stencil import SchemeStencil


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


class StubSpecialist:
    def provide(self, stencil: SchemeStencil):
        coefficients = stencil.coefficients
        fixed_scale = stencil.scale

        if stencil.apply:

            def apply_kernel(
                step: float,
                origin,
                *terms,
            ):
                *translations, result = terms
                result.value = origin.value + step * fixed_scale * sum(
                    coefficient * translation.value
                    for coefficient, translation in zip(
                        coefficients,
                        translations,
                        strict=True,
                    )
                )
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms,
        ):
            *translations, out = terms
            out.value = step * fixed_scale * sum(
                coefficient * translation.value
                for coefficient, translation in zip(coefficients, translations, strict=True)
            )
            return out

        return delta_kernel


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def test_euler_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeEuler.__dict__


def test_euler_default_call_path_is_scheme_owned_inline_call() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarAllocator())

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeEuler.call_inline
    assert scheme.redirect_call == scheme.call_step


def test_euler_public_call_uses_redirect_call() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

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


def test_euler_inline_call_performs_one_forward_euler_step() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)


def test_euler_inline_call_clips_to_remaining_interval() -> None:
    scheme = SchemeEuler(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05)


def test_euler_specialist_path_is_selected_inside_scheme() -> None:
    scheme = SchemeEuler(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeEuler.call_specialized
    assert scheme.redirect_call == scheme.call_step


def test_euler_monitoring_records_fixed_step_without_changing_pure_path() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(exponential_growth, ScalarAllocator(), monitor=monitor.scheme)
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    assert scheme.monitor is monitor.scheme
    assert scheme.call_body.__func__ is SchemeEuler.call_inline
    assert scheme.call_step.__func__ is scheme.call_monitored.__func__
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)
    assert monitor.scheme.adaptive_steps == []
    assert len(monitor.scheme.fixed_steps) == 1

    step = monitor.scheme.fixed_steps[0]
    assert step.scheme == "Euler"
    assert step.t_start == pytest.approx(0.0)
    assert step.t_end == pytest.approx(0.125)
    assert step.accepted_dt == pytest.approx(0.125)



def test_euler_monitoring_records_specialist_fixed_step() -> None:
    monitor = Monitor()
    scheme = SchemeEuler(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
        monitor=monitor.scheme,
    )
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    assert scheme.monitor is monitor.scheme
    assert scheme.call_body.__func__ is SchemeEuler.call_specialized
    assert scheme.call_step.__func__ is scheme.call_monitored.__func__
    assert scheme.redirect_call == scheme.call_step

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.125)
    assert len(monitor.scheme.fixed_steps) == 1
    assert monitor.scheme.fixed_steps[0].scheme == "Euler"


def test_euler_inline_and_specialist_paths_match_for_one_step() -> None:
    interval_inline = Interval(present=0.0, step=0.125, stop=1.0)
    interval_specialist = Interval(present=0.0, step=0.125, stop=1.0)
    state_inline = ScalarState(1.0)
    state_specialist = ScalarState(1.0)

    inline = SchemeEuler(exponential_growth, ScalarAllocator())
    specialist = SchemeEuler(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_specialist = specialist(
        interval_specialist,
        state_specialist,
    )

    assert accepted_dt_inline == pytest.approx(accepted_dt_specialist)
    assert state_inline.value == pytest.approx(state_specialist.value)
    assert state_inline.value == pytest.approx(1.125)
