"""Concrete derivative split declarations for problem definitions."""

from __future__ import annotations

from dataclasses import dataclass

from stark.core.contracts.derivative import DerivativeLike


@dataclass(frozen=True, slots=True)
class DerivativeSplit:
    """
    Concrete implicit-explicit derivative split.

    Use `Derivative.split(implicit=..., explicit=...)` to create this object in
    user code. IMEX schemes consume the protocol shape rather than this
    concrete class, but the problem layer owns this declaration helper because
    it is part of describing the differential problem.

    The full right-hand side is understood as

        f(t, x) = f_implicit(t, x) + f_explicit(t, x)

    where both parts write into translation objects in place.
    """

    implicit: DerivativeLike
    explicit: DerivativeLike

    @property
    def im(self) -> DerivativeLike:
        """Short alias for the implicit derivative worker."""

        return self.implicit

    @property
    def ex(self) -> DerivativeLike:
        """Short alias for the explicit derivative worker."""

        return self.explicit


__all__ = ["DerivativeSplit"]
