"""Shared STARK exception contracts."""

from __future__ import annotations


class StarkError(Exception):
    """Base class for STARK-owned exceptions."""


class StarkErrorRecoverable(StarkError):
    """
    Failure that a caller may handle by rejecting work and trying again.

    Adaptive schemes catch this base class around implicit stage solves. Custom
    resolvents should raise this, or a subclass, when the current trial step is
    allowed to fail without aborting the whole integration.
    """


__all__ = ["StarkError", "StarkErrorRecoverable"]
