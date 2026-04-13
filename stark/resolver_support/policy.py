from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResolverPolicy:
    """Iteration-control policy for nonlinear resolvers."""

    max_iterations: int = 16

    def __repr__(self) -> str:
        return f"ResolverPolicy(max_iterations={self.max_iterations!r})"

    def __str__(self) -> str:
        return f"max_iterations={self.max_iterations}"


__all__ = ["ResolverPolicy"]
