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


class StarkEngine(Protocol):
    """Backend bundle for a declared STARK layout."""

    layout: Any
    algebraist_layout: Any
    accelerator: Accelerator
    allocator: Allocator
    algebraist_general: Any
    algebraist_specialist: Any


__all__ = ["StarkEngine"]
