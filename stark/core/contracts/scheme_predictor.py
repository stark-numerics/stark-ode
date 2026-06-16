from __future__ import annotations

from typing import Protocol

from stark.core.contracts.linear_combine import Scale
from stark.core.contracts.translation import Translation


class SchemePredictorLike(Protocol):
    """Protocol for seeding an implicit stage unknown before resolution.

    Predictors are scheme-layer strategy workers. They write a deliberate
    starting value into ``delta`` before the scheme hands the stage equation to
    a resolvent. Resolvents and inverters remain unaware of prediction policy.
    """

    def __call__(
        self,
        *,
        known: Translation | None,
        previous: Translation | None,
        delta: Translation,
        scale: Scale,
    ) -> Translation:
        """Write the predicted stage increment into ``delta`` and return it."""
        ...


__all__ = ["SchemePredictorLike"]
