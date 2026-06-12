from __future__ import annotations

from stark.core.contracts.errors import StarkErrorRecoverable


class ResolventError(StarkErrorRecoverable):
    """Raised when a nonlinear resolvent fails to resolve its residual."""


__all__ = ["ResolventError"]









