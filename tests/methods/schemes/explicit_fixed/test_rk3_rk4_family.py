from __future__ import annotations

import pytest

from stark import Interval
from stark.methods.schemes.explicit.fixed.kutta3 import SchemeKutta3
from stark.methods.schemes.explicit.fixed.rk38 import SchemeRK38
from stark.methods.schemes.explicit.fixed.ssprk33 import SchemeSSPRK33
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauSpecialist,
    dummy_exponential_growth_rhs,
)


@pytest.mark.parametrize(
    ("scheme_cls", "expected"),
    [
        (SchemeKutta3, 1.1331380208333333),
        (SchemeSSPRK33, 1.1331380208333333),
        (SchemeRK38, 1.133148193359375),
    ],
)
def test_rk3_rk4_scheme_performs_one_step(scheme_cls, expected) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.125)
    assert state.value == pytest.approx(expected)


@pytest.mark.parametrize(
    ("scheme_cls", "expected"),
    [
        (SchemeKutta3, 1.0512708333333334),
        (SchemeSSPRK33, 1.0512708333333334),
        (SchemeRK38, 1.05127109375),
    ],
)
def test_rk3_rk4_scheme_clips_to_remaining_interval(scheme_cls, expected) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

    assert accepted_dt == pytest.approx(0.05)
    assert state.value == pytest.approx(expected)


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeSSPRK33,
        SchemeRK38,
    ],
)
def test_rk3_rk4_default_call_path_is_scheme_owned_inline_call(scheme_cls) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())

    assert scheme.redirect_call == scheme.call_step


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeSSPRK33,
        SchemeRK38,
    ],
)
def test_rk3_rk4_specialist_path_is_selected_inside_scheme(scheme_cls) -> None:
    scheme = scheme_cls(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
    )

    assert scheme.redirect_call == scheme.call_step


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeKutta3,
        SchemeSSPRK33,
        SchemeRK38,
    ],
)
def test_rk3_rk4_inline_and_specialist_paths_match_for_one_step(scheme_cls) -> None:
    interval_inline = Interval(present=0.0, step=0.125, stop=1.0)
    interval_specialist = Interval(present=0.0, step=0.125, stop=1.0)
    state_inline = DummyScalarState(1.0)
    state_specialist = DummyScalarState(1.0)

    inline = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    specialist = scheme_cls(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline)
    accepted_dt_specialist = specialist(
        interval_specialist,
        state_specialist,
    )

    assert accepted_dt_inline == pytest.approx(accepted_dt_specialist)
    assert state_inline.value == pytest.approx(state_specialist.value)
