"""Contracts for translation-space inner products."""

from __future__ import annotations

from typing import Any, Protocol


class InnerProduct(Protocol):
    """
    Return the inner product of two translations.

    Norms alone are not enough for Krylov methods. If a resolvent or inverter
    needs orthogonalization or secant projections, the user must also provide
    an inner product compatible with the translation space.
    """

    def __call__(self, left: Any, right: Any) -> float:
        ...


__all__ = ["InnerProduct"]
