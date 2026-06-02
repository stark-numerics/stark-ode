from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ResolventPolicy:
    """Iteration-control policy for nonlinear resolvents."""

    max_iterations: int = 16

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("ResolventPolicy.max_iterations must be at least 1.")

    def __repr__(self) -> str:
        return f"ResolventPolicy(max_iterations={self.max_iterations!r})"

    def __str__(self) -> str:
        return f"max_iterations={self.max_iterations}"


__all__ = ["ResolventPolicy"]









