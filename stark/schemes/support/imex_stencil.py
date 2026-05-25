from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stark.schemes.support.stencil import SchemeStencil
from stark.schemes.support.tableau import ButcherTableauImex


@dataclass(frozen=True, slots=True)
class SchemeStencilImexTableau:
    """Stencil view of paired explicit/implicit IMEX tableaux.

    Source ordering is explicit derivatives first, then implicit derivatives.
    For a stage row ``i`` this means::

        kE_0, ..., kE_{i-1}, kI_0, ..., kI_{i-1}

    For advance/error rows this means::

        kE_0, ..., kE_{n-1}, kI_0, ..., kI_{n-1}
    """

    tableau: ButcherTableauImex
    scale: float = 1.0

    def stage_rhs(self, index: int) -> SchemeStencil:
        """Known IMEX stage right-hand side, excluding the diagonal implicit term."""

        if not isinstance(index, int):
            raise TypeError("stage index must be an int.")
        if index < 0 or index >= len(self.tableau.implicit.a):
            raise IndexError(
                f"stage index {index} is out of range for IMEX tableau with "
                f"{len(self.tableau.implicit.a)} stage row(s)."
            )

        explicit_row = self.tableau.explicit.a[index]
        implicit_row = self.tableau.implicit.a[index]
        return SchemeStencil(
            (*explicit_row[:index], *implicit_row[:index]),
            scale=self.scale,
            apply=False,
        )

    def advance_delta(self) -> SchemeStencil:
        """High-order accepted IMEX increment from explicit and implicit weights."""

        return SchemeStencil(
            (*self.tableau.explicit.b, *self.tableau.implicit.b),
            scale=self.scale,
            apply=False,
        )

    def error_delta(self) -> SchemeStencil:
        """Embedded IMEX error increment from explicit and implicit weight differences."""

        explicit_error = _error_weights(
            self.tableau.explicit.b_high,
            self.tableau.explicit.b_low,
        )
        implicit_error = _error_weights(
            self.tableau.implicit.b_high,
            self.tableau.implicit.b_low,
        )
        return SchemeStencil((*explicit_error, *implicit_error), scale=self.scale)


def _error_weights(high: Iterable[float], low: Iterable[float]) -> tuple[float, ...]:
    high_tuple = tuple(high)
    low_tuple = tuple(low)
    if len(high_tuple) != len(low_tuple):
        raise ValueError("IMEX embedded weights must have matching lengths.")
    return tuple(
        high_weight - low_weight
        for high_weight, low_weight in zip(high_tuple, low_tuple, strict=True)
    )


__all__ = ["SchemeStencilImexTableau"]
