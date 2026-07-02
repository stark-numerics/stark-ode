from __future__ import annotations

import pytest

from stark import Interval
from stark.core.contracts import IntervalLike
from stark.methods.schemes.explicit.fixed.heun import SchemeHeun
from stark.methods.schemes.explicit.fixed.midpoint import SchemeMidpoint
from stark.methods.schemes.explicit.fixed.ralston import SchemeRalston
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyTableauSpecialist,
    dummy_exponential_growth_rhs,
)


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
def test_rk2_default_call_path_is_scheme_owned_inline_call(scheme_cls) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())

    assert scheme.redirect_call == scheme.call_step


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_public_call_uses_redirect_call(scheme_cls) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    def replacement_call(
        replacement_interval: IntervalLike,
        replacement_state: DummyScalarState,
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
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_call_inline_performs_one_second_order_step(scheme_cls) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

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
def test_rk2_call_inline_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = scheme_cls(dummy_exponential_growth_rhs, DummyScalarAllocator())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = DummyScalarState(1.0)

    accepted_dt = scheme(interval, state)

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
def test_rk2_specialist_path_is_selected_inside_scheme(scheme_cls) -> None:
    scheme = scheme_cls(
        dummy_exponential_growth_rhs,
        DummyScalarAllocator(),
        specialist=DummyTableauSpecialist(),
    )

    assert scheme.redirect_call == scheme.call_step


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeHeun,
        SchemeMidpoint,
        SchemeRalston,
    ],
)
def test_rk2_inline_and_specialist_paths_match_for_one_step(scheme_cls) -> None:
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
    assert state_inline.value == pytest.approx(1.1328125)
