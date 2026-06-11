from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, init=False)
class InverterRelaxationStencil:
    """Fixed-coefficient block algebra request for relaxation kernels."""

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



@dataclass(frozen=True, slots=True, init=False)
class InverterRelaxationStencilUpdate:
    """Translation-space relaxation update stencil.

    Represents the block update

        output <- output + damping * update

    as a delta-space specialist request with two translation sources.
    This is deliberately not an apply/state update stencil: inverter outputs
    are translations, not states.
    """

    damping: float
    coefficients: tuple[float, float]
    scale: float
    apply: bool

    def __init__(self, damping: float) -> None:
        object.__setattr__(self, "damping", float(damping))
        object.__setattr__(self, "coefficients", (1.0, float(damping)))
        object.__setattr__(self, "scale", 1.0)
        object.__setattr__(self, "apply", False)


__all__ = ["InverterRelaxationStencil", "InverterRelaxationStencilUpdate"]
