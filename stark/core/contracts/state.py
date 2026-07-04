"""Contracts and type variables for user state objects.

STARK state objects are deliberately unconstrained. They may be dataclasses,
NumPy arrays, nested structures, or domain-specific mutable containers. The
library learns how to allocate and copy them through the `Allocator` or carrier
contracts rather than through inheritance.
"""

from __future__ import annotations

from typing import Any, TypeVar


State = Any

StateType = TypeVar("StateType", bound=State)
"""Invariant type variable for contracts that both receive and produce states.

Use this when a protocol must preserve the exact state object family across
input and output positions. Allocators and copy helpers are the usual examples:
they create state buffers, accept existing state values, and must not silently
change the concrete state shape.
"""

StateTypeCovariant = TypeVar("StateTypeCovariant", bound=State, covariant=True)
"""Covariant type variable for protocols that only produce state objects.

Use this for read-only or factory-style contracts where returning a more
specific state type is always safe. Most mutable STARK contracts need the
invariant form instead because they also consume state objects.
"""

StateTypeContravariant = TypeVar(
    "StateTypeContravariant",
    bound=State,
    contravariant=True,
)
"""Contravariant type variable for protocols that only consume state objects.

Use this for worker contracts such as dynamics and linearizers. A function
that can accept a broader state shape can stand in wherever a narrower state
consumer is required.
"""


__all__ = [
    "State",
    "StateType",
    "StateTypeCovariant",
    "StateTypeContravariant",
]
