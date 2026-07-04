from __future__ import annotations

from typing import cast

from stark.core.contracts.linear_combine import Scale
from stark.core.contracts.translation import TranslationType


class SchemePredictorKnown:
    """Seed an implicit stage with the known explicit stage shift.

    This is the default predictor for ESDIRK-style stage equations of the form
    ``delta = known + alpha * f(...)``. Schemes using this predictor are
    responsible for passing a known shift; the predictor itself stays branch
    free because it can sit on the implicit-stage hot path.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: TranslationType | None,
        previous: TranslationType | None,
        delta: TranslationType,
        scale: Scale[TranslationType],
    ) -> TranslationType:
        del previous
        return scale(1.0, cast(TranslationType, known), delta)


class SchemePredictorZero:
    """Seed every implicit stage with zero.

    This is useful as a simple baseline and for schemes where the known shift is
    not a helpful nonlinear starting point.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: TranslationType | None,
        previous: TranslationType | None,
        delta: TranslationType,
        scale: Scale[TranslationType],
    ) -> TranslationType:
        del known, previous
        return scale(0.0, delta, delta)


class SchemePredictorPrevious:
    """Seed an implicit stage from the previous stage increment when possible.

    Schemes using this predictor are responsible for passing a previous stage
    increment. This keeps the predictor branch-free on the implicit-stage hot
    path; schemes with optional previous values should choose the predictor at
    construction time or branch in their own stage logic.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: TranslationType | None,
        previous: TranslationType | None,
        delta: TranslationType,
        scale: Scale[TranslationType],
    ) -> TranslationType:
        del known
        return scale(1.0, cast(TranslationType, previous), delta)


__all__ = [
    "SchemePredictorKnown",
    "SchemePredictorPrevious",
    "SchemePredictorZero",
]
