from __future__ import annotations

from stark import Configuration
from stark.core.contracts import LinearCombineScaleLike
from stark.methods.schemes.configuration import SchemeConfigurationDefault
from stark.methods.schemes.predictor import (
    SchemePredictorKnown,
    SchemePredictorPrevious,
    SchemePredictorZero,
)
from tests.support import DummyScalarTranslation


ScalarTranslation = DummyScalarTranslation


def test_scheme_predictor_known_uses_known_shift() -> None:
    known = ScalarTranslation(3.0)
    previous = ScalarTranslation(7.0)
    delta = ScalarTranslation(-1.0)

    result = SchemePredictorKnown()(
        known=known,
        previous=previous,
        delta=delta,
        scale=ScalarTranslation.scale,
    )

    assert result is delta
    assert delta.value == 3.0


def test_scheme_predictor_zero_always_zeroes_delta() -> None:
    delta = ScalarTranslation(8.0)

    SchemePredictorZero()(
        known=ScalarTranslation(2.0),
        previous=ScalarTranslation(5.0),
        delta=delta,
        scale=ScalarTranslation.scale,
    )

    assert delta.value == 0.0


def test_scheme_predictor_previous_prefers_previous_stage() -> None:
    delta = ScalarTranslation(-1.0)

    SchemePredictorPrevious()(
        known=ScalarTranslation(2.0),
        previous=ScalarTranslation(5.0),
        delta=delta,
        scale=ScalarTranslation.scale,
    )

    assert delta.value == 5.0


class DummyCustomPredictor:
    def __call__(
        self,
        *,
        known: ScalarTranslation | None,
        previous: ScalarTranslation | None,
        delta: ScalarTranslation,
        scale: LinearCombineScaleLike[ScalarTranslation],
    ) -> ScalarTranslation:
        del known, previous, scale
        delta.value = 11.0
        return delta


def test_scheme_configuration_default_supplies_known_predictor() -> None:
    assert isinstance(SchemeConfigurationDefault().scheme_predictor, SchemePredictorKnown)


def test_configuration_can_supply_custom_predictor() -> None:
    predictor = DummyCustomPredictor()
    configuration = Configuration(scheme_predictor=predictor)

    assert configuration.scheme_predictor is predictor
