"""Shared engine implementation support.

This package is for backend-building pieces that are genuinely shared by more
than one engine. It is not a backend in its own right; user code should usually
select a concrete engine package such as `stark.engines.numpy`.
"""

from stark.engines.shared.basis import EngineTranslationBasis

__all__ = ["EngineTranslationBasis"]
