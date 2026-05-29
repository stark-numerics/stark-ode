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
StateTypeCovariant = TypeVar("StateTypeCovariant", bound=State, covariant=True)
StateTypeContravariant = TypeVar(
    "StateTypeContravariant",
    bound=State,
    contravariant=True,
)


__all__ = [
    "State",
    "StateType",
    "StateTypeCovariant",
    "StateTypeContravariant",
]
