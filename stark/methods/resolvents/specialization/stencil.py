from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True, init=False)
class ResolventStencilBlock:
    """Fixed-coefficient block algebra request for resolvent kernels."""

    coefficients: tuple[float, ...]
    scale: float
    apply: bool

    def __init__(
        self,
        coefficients: Iterable[float],
        *,
        scale: float = 1.0,
        apply: bool = False,
    ) -> None:
        object.__setattr__(
            self,
            "coefficients",
            tuple(float(coefficient) for coefficient in coefficients),
        )
        object.__setattr__(self, "scale", float(scale))
        object.__setattr__(self, "apply", bool(apply))

    @property
    def operation(self) -> Literal["linear_fixed"]:
        return "linear_fixed"


__all__ = ["ResolventStencilBlock"]
