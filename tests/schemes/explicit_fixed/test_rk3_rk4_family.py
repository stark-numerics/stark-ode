from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval
from stark.schemes.explicit.fixed.kutta3 import SchemeKutta3
from stark.schemes.explicit.fixed.rk38 import SchemeRK38
from stark.schemes.explicit.fixed.ssprk33 import SchemeSSPRK33
from stark.schemes.specialization.stencil import SchemeStencil


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


@pytest.mark.parametrize(
    ("scheme_cls", "expected"),
    [
        (SchemeKutta3, 1.1331380208333333),
        (SchemeSSPRK33, 1.1331380208333333),
        (SchemeRK38, 1.133148193359375),
    ],
)
def test_rk3_rk4_scheme_performs_one_step(scheme_cls, expected) -> None:
    scheme = scheme_cls(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.0, step=0.125, stop=1.0)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

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
    scheme = scheme_cls(exponential_growth, ScalarAllocator())
    interval = Interval(present=0.2, step=0.125, stop=0.25)
    state = ScalarState(1.0)

    accepted_dt = scheme(interval, state, Executor())

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
    scheme = scheme_cls(exponential_growth, ScalarAllocator())

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is scheme_cls.call_inline
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
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    assert scheme.call_step.__self__ is scheme
    assert scheme.call_step.__func__ is scheme_cls.call_specialized
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
    state_inline = ScalarState(1.0)
    state_specialist = ScalarState(1.0)

    inline = scheme_cls(exponential_growth, ScalarAllocator())
    specialist = scheme_cls(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    accepted_dt_inline = inline(interval_inline, state_inline, Executor())
    accepted_dt_specialist = specialist(
        interval_specialist,
        state_specialist,
        Executor(),
    )

    assert accepted_dt_inline == pytest.approx(accepted_dt_specialist)
    assert state_inline.value == pytest.approx(state_specialist.value)
