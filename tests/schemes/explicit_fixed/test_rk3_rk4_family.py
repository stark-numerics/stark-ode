from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stark import Executor, Interval
from stark.schemes.explicit_fixed.kutta3 import SchemeKutta3
from stark.schemes.explicit_fixed.rk38 import SchemeRK38
from stark.schemes.explicit_fixed.ssprk33 import SchemeSSPRK33


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

        def solution_state(
            result: ScalarState,
            origin: ScalarState,
            dt: float,
            *rates: ScalarTranslation,
        ) -> None:
            result.value = origin.value + dt * sum(
                weight * rate.value
                for weight, rate in zip(tableau.b, rates, strict=True)
            )

        stages = tuple(
            None if index == 0 else make_stage(index)
            for index in range(len(tableau.c))
        )

        return SimpleNamespace(
            stages=stages,
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
        SchemeKutta3,
        SchemeRK38,
        SchemeSSPRK33,
    ],
)
def test_rk3_rk4_scheme_owns_its_public_call_method(scheme_cls) -> None:
    assert "__call__" in scheme_cls.__dict__


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeRK38,
        SchemeSSPRK33,
    ],
)
def test_rk3_rk4_default_call_path_is_scheme_owned_generic_call(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())

    assert scheme.pure_call.__self__ is scheme
    assert scheme.pure_call.__func__ is scheme_cls.generic_call
    assert scheme.redirect_call == scheme.pure_call


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeRK38,
        SchemeSSPRK33,
    ],
)
def test_rk3_rk4_public_call_uses_redirect_call(scheme_cls) -> None:
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
    ("scheme_cls", "expected"),
    [
        (SchemeKutta3, 1.1331380208333333),
        (SchemeRK38, 1.133148193359375),
        (SchemeSSPRK33, 1.1331380208333333),
    ],
)
def test_rk3_rk4_generic_call_performs_one_step(
    scheme_cls,
    expected: float,
) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("scheme_cls", "expected"),
    [
        (SchemeKutta3, 1.0512708333333334),
        (SchemeRK38, 1.05127109375),
        (SchemeSSPRK33, 1.0512708333333334),
    ],
)
def test_rk3_rk4_generic_call_clips_to_remaining_interval(
    scheme_cls,
    expected: float,
) -> None:
    scheme = scheme_cls(exponential_growth, ScalarWorkbench())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeRK38,
        SchemeSSPRK33,
    ],
)
def test_rk3_rk4_algebraist_path_is_selected_inside_scheme(scheme_cls) -> None:
    scheme = scheme_cls(
        exponential_growth,
        ScalarWorkbench(),
        algebraist=StubAlgebraist(),
    )

    assert scheme.pure_call.__self__ is scheme
    assert scheme.pure_call.__func__ is scheme_cls.algebraist_call
    assert scheme.redirect_call == scheme.pure_call


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeRK38,
        SchemeSSPRK33,
    ],
)
def test_rk3_rk4_generic_and_algebraist_paths_match_for_one_step(
    scheme_cls,
) -> None:
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