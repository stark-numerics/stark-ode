"""Protocol for backend engines used by the high-level interface.

Engines bundle backend choices for a declared frame. They do not describe the
differential equation or numerical method; they expose allocation,
acceleration, and generator resources that a system/method construction path
can use when binding a concrete problem.
"""

from __future__ import annotations

from typing import Any, Protocol

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.allocator import AllocatorLike


class Engine(Protocol):
    """Backend bundle for a declared STARK frame."""

    @property
    def frame(self) -> Any:
        """Frame-like layout used to construct this backend bundle."""
        ...

    @property
    def accelerator(self) -> Accelerator:
        """Accelerator used for generated or user-supplied kernels."""
        ...

    @property
    def allocator(self) -> AllocatorLike:
        """Allocator for backend-owned state and translation objects."""
        ...

    @property
    def carriers(self) -> tuple[Any, ...]:
        """Carrier objects corresponding to the frame fields."""
        ...

    @property
    def generator(self) -> Any:
        """Prepared generator facade for backend/frame code generation."""
        ...

    def translation_basis(self) -> Any:
        """Coordinate basis for engine-owned translations.

        This is an advanced inspection/materialisation hook. Ordinary problem
        declarations should not need it; dense inverters and diagnostics can
        use it when they need coordinates for backend translation objects.
        """
        ...


__all__ = ["Engine"]
