from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from stark.schemes.support.tableau import ButcherTableau


SchemeStencilCoefficient: TypeAlias = float


@dataclass(frozen=True, slots=True, init=False)
class SchemeStencil:
    """Fixed-coefficient scheme stencil.

    A stencil describes the fixed coefficient part of an operation such as:

        step * scale * sum(c_i * source_i)

    The runtime step is supplied later to the kernel produced by an Algebraist
    specialist. The coefficients and fixed scale are supplied here.
    """

    coefficients: tuple[SchemeStencilCoefficient, ...]
    scale: float

    def __init__(
        self,
        coefficients: Iterable[float],
        *,
        scale: float = 1.0,
    ) -> None:
        object.__setattr__(
            self,
            "coefficients",
            tuple(float(coefficient) for coefficient in coefficients),
        )
        object.__setattr__(self, "scale", float(scale))


@dataclass(frozen=True, slots=True)
class SchemeStencilTableau:
    """Stencil view of a Butcher tableau.

    Stage indexing is zero-based and follows ``tableau.a[index]`` directly.
    For example, ``stage(1)`` returns the second tableau row.
    """

    tableau: ButcherTableau
    scale: float = 1.0

    def stage(self, index: int) -> SchemeStencil:
        """Return the stencil for a tableau stage row."""

        if not isinstance(index, int):
            raise TypeError("stage index must be an int.")

        if index < 0 or index >= len(self.tableau.a):
            raise IndexError(
                f"stage index {index} is out of range for tableau with "
                f"{len(self.tableau.a)} stage row(s)."
            )

        return SchemeStencil(
            self.tableau.a[index],
            scale=self.scale,
        )

    def advance(self) -> SchemeStencil:
        """Return the stencil for the accepted step advance weights."""

        return SchemeStencil(
            self.tableau.b,
            scale=self.scale,
        )

    def error(self) -> SchemeStencil:
        """Return the stencil for embedded error weights."""

        if self.tableau.b_embedded is None:
            raise ValueError(
                "Cannot build an error stencil from a tableau without "
                "embedded weights."
            )

        high = self.tableau.b_high
        low = self.tableau.b_low

        if len(high) != len(low):
            raise ValueError(
                "Cannot build an error stencil because high-order and "
                "low-order weights have different lengths."
            )

        return SchemeStencil(
            (high_weight - low_weight for high_weight, low_weight in zip(high, low)),
            scale=self.scale,
        )


__all__ = [
    "SchemeStencil",
    "SchemeStencilCoefficient",
    "SchemeStencilTableau",
]