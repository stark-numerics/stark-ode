from __future__ import annotations

from stark.core.contracts import SchemePredictorLike, Translation
from stark.core.contracts.linear_combine import Scale
from stark.methods.schemes.configuration import SchemeConfiguration


class SchemePredictorKnown:
    """Seed an implicit stage with the known explicit stage shift.

    This is the default predictor for ESDIRK-style stage equations of the form
    ``delta = known + alpha * f(...)``. If a scheme has no known shift for a
    particular stage, the predictor writes zero into ``delta`` rather than
    leaving an accidental previous value in place.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: Translation | None,
        previous: Translation | None,
        delta: Translation,
        scale: Scale,
    ) -> Translation:
        del previous
        if known is None:
            return scale(0.0, delta, delta)
        return scale(1.0, known, delta)


class SchemePredictorZero:
    """Seed every implicit stage with zero.

    This is useful as a simple baseline and for schemes where the known shift is
    not a helpful nonlinear starting point.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: Translation | None,
        previous: Translation | None,
        delta: Translation,
        scale: Scale,
    ) -> Translation:
        del known, previous
        return scale(0.0, delta, delta)


class SchemePredictorPrevious:
    """Seed an implicit stage from the previous stage increment when possible.

    If no previous increment is available, this falls back to the known shift; if
    neither is available, it writes zero. The predictor is deliberately total so
    an implicit stage never starts from stale buffer contents.
    """

    __slots__ = ()

    def __call__(
        self,
        *,
        known: Translation | None,
        previous: Translation | None,
        delta: Translation,
        scale: Scale,
    ) -> Translation:
        if previous is not None:
            return scale(1.0, previous, delta)
        if known is not None:
            return scale(1.0, known, delta)
        return scale(0.0, delta, delta)


def resolve_scheme_predictor(
    configuration: SchemeConfiguration | None,
) -> SchemePredictorLike:
    """Return the configured scheme predictor or the package default."""

    if configuration is not None and configuration.scheme_predictor is not None:
        return configuration.scheme_predictor
    return SchemePredictorKnown()


__all__ = [
    "SchemePredictorKnown",
    "SchemePredictorPrevious",
    "SchemePredictorZero",
    "resolve_scheme_predictor",
]
