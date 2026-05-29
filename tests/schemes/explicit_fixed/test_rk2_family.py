from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Executor, Interval
from stark.schemes.explicit_fixed.heun import SchemeHeun
from stark.schemes.explicit_fixed.midpoint import SchemeMidpoint
from stark.schemes.explicit_fixed.ralston import SchemeRalston
from stark.schemes.support.stencil import SchemeStencil


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
    scheme = scheme_cls(exponential_growth, ScalarAllocator())

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_inline
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
    scheme = scheme_cls(exponential_growth, ScalarAllocator())
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
def test_rk2_call_inline_performs_one_second_order_step(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarAllocator())
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
def test_rk2_call_inline_clips_to_remaining_interval(scheme_cls) -> None:
    scheme = scheme_cls(exponential_growth, ScalarAllocator())
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
def test_rk2_specialist_path_is_selected_inside_scheme(scheme_cls) -> None:
    scheme = scheme_cls(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    assert scheme.call_pure.__self__ is scheme
    assert scheme.call_pure.__func__ is scheme_cls.call_specialized
    assert scheme.redirect_call == scheme.call_pure


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
    assert state_inline.value == pytest.approx(1.1328125)
