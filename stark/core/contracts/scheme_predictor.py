from __future__ import annotations

from typing import Protocol

from stark.core.contracts.linear_combine import Scale
from stark.core.contracts.translation import TranslationType


class SchemePredictorLike(Protocol[TranslationType]):
    """Protocol for seeding an implicit stage unknown before resolution.

    Predictors are scheme-layer strategy workers. They write a deliberate
    starting value into ``delta`` before the scheme hands the stage equation to
    a resolvent. Resolvents and inverters remain unaware of prediction policy.

    The predictor is generic in the translation it seeds so callers keep their
    concrete scalar, vector, backend, or test translation type through the
    branch-free hot path.
    """

    def __call__(
        self,
        *,
        known: TranslationType | None,
        previous: TranslationType | None,
        delta: TranslationType,
        scale: Scale[TranslationType],
    ) -> TranslationType:
        """Write the predicted stage increment into ``delta`` and return it."""
        ...


__all__ = ["SchemePredictorLike"]
