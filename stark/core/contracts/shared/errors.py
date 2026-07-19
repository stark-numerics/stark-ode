"""Shared STARK exception contracts."""

from __future__ import annotations


class StarkError(Exception):
    """Base class for STARK-owned exceptions."""


class StarkErrorRecoverable(StarkError):
    """
    Failure that a caller may handle by rejecting work and trying again.
    """

class StarkErrorUnrecoverable(StarkError):
    """
    Failure that a caller cannot handle by rejecting work and trying again.
    """



__all__ = ["StarkError", "StarkErrorRecoverable", "StarkErrorUnrecoverable"]
