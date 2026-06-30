from __future__ import annotations

from dataclasses import dataclass

from stark import Configuration
from stark.methods.schemes.configuration import SchemeConfigurationDefault
from stark.methods.schemes.predictor import (
    SchemePredictorKnown,
    SchemePredictorPrevious,
    SchemePredictorZero,
)


@dataclass(slots=True)
class TranslationScalar:
    value: float = 0.0


def scale(a: float, x: TranslationScalar, out: TranslationScalar) -> TranslationScalar:
    out.value = a * x.value
    return out


def test_scheme_predictor_known_uses_known_shift() -> None:
    known = TranslationScalar(3.0)
    previous = TranslationScalar(7.0)
    delta = TranslationScalar(-1.0)

    result = SchemePredictorKnown()(known=known, previous=previous, delta=delta, scale=scale)

    assert result is delta
    assert delta.value == 3.0


def test_scheme_predictor_zero_always_zeroes_delta() -> None:
    delta = TranslationScalar(8.0)

    SchemePredictorZero()(known=TranslationScalar(2.0), previous=TranslationScalar(5.0), delta=delta, scale=scale)

    assert delta.value == 0.0


def test_scheme_predictor_previous_prefers_previous_stage() -> None:
    delta = TranslationScalar(-1.0)

    SchemePredictorPrevious()(known=TranslationScalar(2.0), previous=TranslationScalar(5.0), delta=delta, scale=scale)

    assert delta.value == 5.0


class CustomPredictor:
    def __call__(self, *, known, previous, delta, scale):
        del known, previous, scale
        delta.value = 11.0
        return delta


def test_scheme_configuration_default_supplies_known_predictor() -> None:
    assert isinstance(SchemeConfigurationDefault().scheme_predictor, SchemePredictorKnown)


def test_configuration_can_supply_custom_predictor() -> None:
    predictor = CustomPredictor()
    configuration = Configuration(scheme_predictor=predictor)

    assert configuration.scheme_predictor is predictor
