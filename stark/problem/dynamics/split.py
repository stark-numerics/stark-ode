"""Concrete dynamics split declarations for problem definitions."""

from __future__ import annotations

from dataclasses import dataclass

from stark.core.contracts.problem.dynamics import DynamicsLike


@dataclass(frozen=True, slots=True)
class DynamicsSplit:
    """
    Concrete implicit-explicit dynamics split.

    Use `Dynamics.split(implicit=..., explicit=...)` to create this object in
    user code. IMEX schemes consume the protocol shape rather than this
    concrete class, but the problem layer owns this declaration helper because
    it is part of describing the differential problem.

    The full right-hand side is understood as

        f(t, x) = f_implicit(t, x) + f_explicit(t, x)

    where both parts write into translation objects in place.
    """

    implicit: DynamicsLike
    explicit: DynamicsLike

__all__ = ["DynamicsSplit"]
