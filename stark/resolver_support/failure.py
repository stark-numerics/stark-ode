from __future__ import annotations


class ResolutionError(RuntimeError):
    """Raised when a nonlinear resolver fails to resolve its residual."""


__all__ = ["ResolutionError"]
