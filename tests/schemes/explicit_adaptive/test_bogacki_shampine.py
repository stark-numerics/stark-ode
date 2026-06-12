from __future__ import annotations

from dataclasses import dataclass

from stark import Interval, Tolerance


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

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


class StubSpecialist:
    def provide(self, stencil):
        coefficients = tuple(stencil.coefficients)
        stencil_scale = stencil.scale

        if stencil.apply:
            def apply_kernel(
                step: float,
                origin,
                *terms,
            ):
                *translations, result = terms
                delta = _combine_delta(step, stencil_scale, coefficients, translations)
                delta(origin, result)
                return result

            return apply_kernel

        def delta_kernel(
            step: float,
            *terms,
        ):
            *translations, out = terms
            delta = _combine_delta(step, stencil_scale, coefficients, translations)
            out.value = delta.value
            return out

        return delta_kernel


def _combine_delta(
    step: float,
    stencil_scale: float,
    coefficients: tuple[float, ...],
    translations: tuple[ScalarTranslation, ...],
) -> ScalarTranslation:
    if len(coefficients) != len(translations):
        raise AssertionError(
            f"stencil arity {len(coefficients)} received "
            f"{len(translations)} translation(s)"
        )

    if not translations:
        return ScalarTranslation()

    total = 0.0 * translations[0]
    for coefficient, translation in zip(coefficients, translations, strict=True):
        total = total + (step * stencil_scale * coefficient) * translation
    return total


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


def tight_configuration() -> Configuration:
    return Configuration(scheme_tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


import pytest

from stark import Integrator, Interval, IntegratorStepper
from stark.diagnostics.monitor import Monitor
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine


def test_bogacki_shampine_owns_its_public_call_method() -> None:
    assert "__call__" in SchemeBogackiShampine.__dict__


def test_bogacki_shampine_default_advance_path_is_scheme_owned_inline_advance() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeBogackiShampine.call_inline
    assert scheme.redirect_call.__self__ is scheme
    assert scheme.redirect_call.__func__ is scheme.call_step.__func__


def test_bogacki_shampine_public_call_uses_redirect_call() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState
    ) -> float:
        del replacement_interval
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


def test_bogacki_shampine_call_returns_accepted_dt_and_updates_next_step() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_call_clips_to_remaining_interval() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_specialist_path_is_selected_inside_scheme() -> None:
    scheme = SchemeBogackiShampine(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is SchemeBogackiShampine.call_specialized


def test_bogacki_shampine_inline_and_specialist_paths_match_for_one_step() -> None:
    interval_inline = Interval(present=0.0, step=0.1, stop=0.3)
    interval_specialist = Interval(present=0.0, step=0.1, stop=0.3)
    state_inline = ScalarState(1.0)
    state_specialist = ScalarState(1.0)

    inline = SchemeBogackiShampine(exponential_growth, ScalarAllocator())
    specialist = SchemeBogackiShampine(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_specialist = specialist(
        interval_specialist,
        state_specialist,
    )

    assert accepted_dt_specialist == pytest.approx(accepted_dt_inline)
    assert state_specialist.value == pytest.approx(state_inline.value)
    assert interval_specialist.step == pytest.approx(interval_inline.step)


def test_bogacki_shampine_integration_matches_characterized_step_count() -> None:
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator())
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    outputs = list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(outputs) == 2
    assert interval.present == pytest.approx(0.3)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


def test_bogacki_shampine_monitoring_records_existing_adaptive_fields() -> None:
    monitor = Monitor()
    scheme = SchemeBogackiShampine(zero_rhs, ScalarAllocator(), monitor=monitor.scheme)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    list(Integrator().mutating_trajectory(stepper, interval, state))

    assert len(monitor.scheme.adaptive_steps) == 2
    first = monitor.scheme.adaptive_steps[0]
    second = monitor.scheme.adaptive_steps[1]

    assert first.scheme == "BS23"
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio == pytest.approx(0.0)
    assert first.rejection_count == 0

    assert second.scheme == "BS23"
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio == pytest.approx(0.0)
    assert second.rejection_count == 0
