"""Protocol for backend engines used by the high-level interface.

Engines bundle backend choices for a declared layout. They do not describe the
differential equation or numerical method; they expose the allocation,
acceleration, and algebra providers that a system/method construction path can
use when binding a concrete problem.
"""

from __future__ import annotations

from typing import Any, Protocol

from stark.contracts.accelerator import Accelerator
from stark.contracts.allocator import Allocator


class Engine(Protocol):
    """Backend bundle for a declared STARK layout."""

    @property
    def layout(self) -> Any:
        """User-facing layout used to construct this backend bundle."""
        ...

    @property
    def algebraist_layout(self) -> Any:
        """Algebraist layout derived from the user-facing layout."""
        ...

    @property
    def accelerator(self) -> Accelerator:
        """Accelerator used for generated or user-supplied kernels."""
        ...

    @property
    def allocator(self) -> Allocator:
        """Allocator for backend-owned state and translation objects."""
        ...

    @property
    def carriers(self) -> tuple[Any, ...]:
        """Carrier objects corresponding to the algebraist layout fields."""
        ...

    @property
    def algebraist_linear_combine(self) -> Any:
        """Provider for backend linear-combine kernels."""
        ...

    @property
    def algebraist_norm(self) -> Any:
        """Provider for backend norm kernels."""
        ...

    @property
    def algebraist_inner_product(self) -> Any:
        """Provider for backend inner-product kernels."""
        ...

    @property
    def algebraist_specialist(self) -> Any:
        """Provider for backend specialist kernels."""
        ...


__all__ = ["Engine"]
