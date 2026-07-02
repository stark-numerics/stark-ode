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
    def provide_delta(self, stencil):
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

    provide_apply = provide_delta


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

from stark import Interval
from stark.methods.schemes.explicit.adaptive.bogacki_shampine import SchemeBogackiShampine
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.explicit.adaptive.dormand_prince import SchemeDormandPrince
from stark.methods.schemes.explicit.adaptive.fehlberg45 import SchemeFehlberg45
from stark.methods.schemes.explicit.adaptive.tsitouras5 import SchemeTsitouras5


@pytest.mark.parametrize(
    "scheme_cls",
    [
        SchemeBogackiShampine,
        SchemeCashKarp,
        SchemeDormandPrince,
        SchemeFehlberg45,
        SchemeTsitouras5,
    ],
)
def test_explicit_adaptive_specialist_path_matches_inline_path(scheme_cls) -> None:
    interval_inline = Interval(present=0.0, step=0.1, stop=0.3)
    interval_specialist = Interval(present=0.0, step=0.1, stop=0.3)
    state_inline = ScalarState(1.0)
    state_specialist = ScalarState(1.0)

    inline = scheme_cls(exponential_growth, ScalarAllocator())
    specialized = scheme_cls(
        exponential_growth,
        ScalarAllocator(),
        specialist=StubSpecialist(),
    )

    accepted_inline = inline(interval_inline, state_inline)
    accepted_specialist = specialized(
        interval_specialist,
        state_specialist,
    )

    assert accepted_specialist == pytest.approx(accepted_inline)
    assert state_specialist.value == pytest.approx(state_inline.value)
    assert interval_specialist.step == pytest.approx(interval_inline.step)

