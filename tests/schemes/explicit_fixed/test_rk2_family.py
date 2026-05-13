from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Interval
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.midpoint import SchemeMidpoint
from stark.schemes.explicit_fixed.ralston import SchemeRalston


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
        def stage2(
            stage: ScalarState,
            state: ScalarState,
            dt: float,
            k1: ScalarTranslation,
        ) -> None:
            stage.value = state.value + dt * tableau.a[1][0] * k1.value

        def solution_state(
            result: ScalarState,
            origin: ScalarState,
            dt: float,
            *rates: ScalarTranslation,
        ) -> None:
            if len(rates) == 1:
                result.value = origin.value + dt * rates[0].value
                return

            result.value = origin.value + dt * sum(
                weight * rate.value
                for weight, rate in zip(tableau.b, rates, strict=True)
            )

        return SimpleNamespace(
            stages=(None, stage2),
            solution_state=solution_state,
        )


def exponential_growth(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval
    out.value = state.value


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_default_call_path_is_scheme_owned_call_generic(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_generic
    assert scheme.redirect_call == scheme.call_pure


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

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
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_call_generic_performs_one_second_order_step(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(1.1328125)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_call_generic_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(1.05125)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_algebraist_path_is_selected_inside_scheme(scheme_cls) -> None:
    scheme = scheme_cls(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.algebraist_call
    assert scheme.redirect_call == scheme.call_pure


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_generic_and_algebraist_paths_match_for_one_step(scheme_cls) -> None:
    interval_generic = Interval(present=0.0, step=0.125, stop=1.0)
    interval_algebraist = Interval(present=0.0, step=0.125, stop=1.0)
    state_generic = ScalarState(1.0)
    state_algebraist = ScalarState(1.0)

    generic = scheme_cls(exponential_growth, ScalarWorkbench())
    algebraist = scheme_cls(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    accepted_dt_generic = generic(interval_generic, state_generic, Executor())
    accepted_dt_algebraist = algebraist(
        interval_algebraist,
        state_algebraist,
        Executor(),
    )

    assert accepted_dt_generic == pytest.approx(accepted_dt_algebraist)
    assert state_generic.value == pytest.approx(state_algebraist.value)
    assert state_generic.value == pytest.approx(1.1328125)