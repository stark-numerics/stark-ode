"""Contracts for frame-like structured state declarations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from stark.core.contracts.problem.field import FieldLike, FieldPath
from stark.core.contracts.problem.inner_product import InnerProductNamed
from stark.core.contracts.problem.norm import NormLike


class FrameLike(Protocol):
    """Structural contract consumed by frame-aware engines.

    A frame-like object owns ordered field declarations plus the matching
    per-field norm and inner-product policies. Concrete frame implementations
    may carry richer metadata, but algebra machinery should only depend on
    these canonical collections.
    """

    @property
    def fields(self) -> Sequence[FieldLike[Any, Any]]:
        """Ordered fields visible to allocation and algebra kernels."""
        ...

    @property
    def norms(self) -> Sequence[NormLike[Any]]:
        """Per-field norm policies in the same order as `fields`."""
        ...

    @property
    def inner_products(self) -> Sequence[InnerProductNamed[Any]]:
        """Per-field inner-product policies in the same order as `fields`."""
        ...

    @property
    def translation_paths(self) -> Sequence[FieldPath]:
        """Translation-side paths in frame order."""
        ...

    @property
    def state_paths(self) -> Sequence[FieldPath]:
        """State-side paths in frame order."""
        ...


__all__ = ["FrameLike"]
