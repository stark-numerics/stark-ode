from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Integrator, Interval, Marcher, Tolerance
from stark.monitor import Monitor
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.explicit_adaptive.fehlberg45 import SchemeFehlberg45


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


class StubAlgebraist:
    def bind_explicit_scheme(self, tableau):
        high_indices = tuple(index for index, weight in enumerate(tableau.b) if weight != 0.0)
        low = tableau.b_embedded
        assert low is not None

        error_weights = tuple(
            high - embedded for high, embedded in zip(tableau.b, low, strict=True)
        )
        error_indices = tuple(
            index for index, weight in enumerate(error_weights) if weight != 0.0
        )

        def make_stage(stage_index: int):
            weights = tableau.a[stage_index]

            def stage_call(
                stage: ScalarState,
                state: ScalarState,
                dt: float,
                *rates: ScalarTranslation,
            ) -> None:
                stage.value = state.value + dt * sum(
                    weight * rate.value
                    for weight, rate in zip(weights, rates, strict=True)
                )

            return stage_call

        def solution(
            out: ScalarTranslation,
            dt: float,
            *rates: ScalarTranslation,
        ) -> ScalarTranslation:
            out.value = dt * sum(
                tableau.b[index] * rate.value
                for index, rate in zip(high_indices, rates, strict=True)
            )
            return out

        def error(
            out: ScalarTranslation,
            dt: float,
            *rates: ScalarTranslation,
        ) -> ScalarTranslation:
            out.value = dt * sum(
                error_weights[index] * rate.value
                for index, rate in zip(error_indices, rates, strict=True)
            )
            return out

        stages = tuple(
            None if index == 0 else make_stage(index)
            for index in range(len(tableau.c))
        )

        return SimpleNamespace(
            stage_state_calls=stages,
            require_stage_state_call=lambda index, scheme_name: stages[index],
            solution_delta_call=solution,
            error_delta_call=error,
            require_error_delta_call=lambda scheme_name: error,
        )


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


def tight_executor() -> Executor:
    return Executor(tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9))


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_default_call_path_is_scheme_owned_generic_call(
    scheme_cls,
) -> None:
    scheme = scheme_cls(zero_rhs, ScalarWorkbench())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_generic


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = scheme_cls(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    def replacement_call(
        replacement_interval: Interval,
        replacement_state: ScalarState,
        replacement_executor: Executor,
    ) -> float:
        del replacement_interval, replacement_executor
        replacement_state.value = 42.0
        return 0.03125

    scheme.redirect_call = replacement_call

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.03125)
    assert state.value == pytest.approx(42.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_call_returns_accepted_dt_and_updates_next_step(
    scheme_cls,
) -> None:
    scheme = scheme_cls(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.1)
    assert interval.step == pytest.approx(0.2)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_call_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = scheme_cls(zero_rhs, ScalarWorkbench())
    interval = Interval(present=0.1, step=1.0, stop=0.3)
    state = ScalarState(2.0)

    accepted_dt = scheme(interval, state, tight_executor())

    assert accepted_dt == pytest.approx(0.2)
    assert interval.step == pytest.approx(0.0)
    assert state.value == pytest.approx(2.0)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_algebraist_path_is_selected_inside_scheme(
    scheme_cls,
) -> None:
    scheme = scheme_cls(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_algebraist


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeCashKarp,
        SchemeFehlberg45,
    ],
)
def test_cash_karp_fehlberg_generic_and_algebraist_paths_match_for_one_step(
    scheme_cls,
) -> None:
    interval_generic = Interval(present=0.0, step=0.1, stop=0.3)
    interval_algebraist = Interval(present=0.0, step=0.1, stop=0.3)
    state_generic = ScalarState(1.0)
    state_algebraist = ScalarState(1.0)

    generic = scheme_cls(exponential_growth, ScalarWorkbench())
    algebraist = scheme_cls(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    accepted_dt_generic = generic(interval_generic, state_generic, tight_executor())
    accepted_dt_algebraist = algebraist(
        interval_algebraist,
        state_algebraist,
        tight_executor(),
    )

    assert accepted_dt_generic == pytest.approx(accepted_dt_algebraist)
    assert state_generic.value == pytest.approx(state_algebraist.value)
    assert interval_generic.step == pytest.approx(interval_algebraist.step)


@pytest.mark.parametrize(
    ("scheme_cls", "scheme_name"),
    [
        (SchemeCashKarp, "RKCK"),
        (SchemeFehlberg45, "RKF45"),
    ],
)
def test_cash_karp_fehlberg_monitoring_records_existing_adaptive_fields(
    scheme_cls,
    scheme_name: str,
) -> None:
    scheme = scheme_cls(zero_rhs, ScalarWorkbench())
    marcher = Marcher(scheme, tight_executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState(2.0)
    monitor = Monitor()

    list(Integrator().live_monitored(marcher, interval, state, monitor))

    assert len(monitor.steps) == 2

    first = monitor.steps[0]
    second = monitor.steps[1]

    assert first.scheme == scheme_name
    assert first.t_start == pytest.approx(0.0)
    assert first.t_end == pytest.approx(0.1)
    assert first.proposed_dt == pytest.approx(0.1)
    assert first.accepted_dt == pytest.approx(0.1)
    assert first.next_dt == pytest.approx(0.2)
    assert first.error_ratio == pytest.approx(0.0)
    assert first.rejection_count == 0

    assert second.scheme == scheme_name
    assert second.t_start == pytest.approx(0.1)
    assert second.t_end == pytest.approx(0.3)
    assert second.proposed_dt == pytest.approx(0.2)
    assert second.accepted_dt == pytest.approx(0.2)
    assert second.next_dt == pytest.approx(0.0)
    assert second.error_ratio == pytest.approx(0.0)
    assert second.rejection_count == 0