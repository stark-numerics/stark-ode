from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

from stark.methods.schemes.method.tableau import Tableau


SchemeStencilCoefficient: TypeAlias = float


@dataclass(frozen=True, slots=True, init=False)
class SchemeStencil:
    """Fixed-coefficient scheme stencil.

    A stencil describes the fixed coefficient part of an operation:

        step * scale * sum(c_i * source_i)

    If ``apply`` is false, a specialist should produce a delta kernel:

        out = step * scale * sum(c_i * source_i)

    If ``apply`` is true, a specialist should produce an apply/update kernel:

        result = origin + step * scale * sum(c_i * source_i)

    The runtime step is supplied later to the produced kernel. The coefficients,
    fixed scale, and operation kind are supplied here.
    """

    coefficients: tuple[SchemeStencilCoefficient, ...]
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


@dataclass(frozen=True, slots=True)
class SchemeStencilTableau:
    """Stencil view of a Butcher tableau.

    Stage indexing is zero-based and follows ``tableau.a[index]`` directly.
    For example, ``stage(1)`` returns the second tableau row.
    """

    tableau: Tableau
    scale: float = 1.0

    def stage(self, index: int) -> SchemeStencil:
        """Return the applied stencil for a tableau stage row.

        A stage row constructs a stage state from the origin state plus a
        weighted sum of already-computed derivative translations.
        """

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
            apply=True,
        )

    def advance_delta(self) -> SchemeStencil:
        """Return the non-applied stencil for accepted advance weights."""

        return SchemeStencil(
            self.tableau.b,
            scale=self.scale,
            apply=False,
        )

    def advance_update(self) -> SchemeStencil:
        """Return the applied stencil for accepted advance weights."""

        return SchemeStencil(
            self.tableau.b,
            scale=self.scale,
            apply=True,
        )

    def error_delta(self) -> SchemeStencil:
        """Return the non-applied stencil for embedded error weights."""

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
            apply=False,
        )


@dataclass(frozen=True, slots=True)
class SchemeStageIncrementStencils:
    """Stencil view for ESDIRK schemes solved in a stage-increment basis.

    Sequential ESDIRK implementations do not store raw ``h * k_i`` rates after
    each implicit solve. They store stage increments:

        delta_i = h * sum(a_ij * k_j)

    with an artificial first increment ``delta_0 = gamma * h * k_0`` for the
    explicit first stage. These stencils convert tableau weights into that
    increment basis once at import time.
    """

    known_shifts: tuple[tuple[float, ...], ...]
    high_delta: tuple[float, ...]
    low_delta: tuple[float, ...]
    error_delta: tuple[float, ...]


def esdirk_stage_increment_stencils(
    tableau: Tableau,
    gamma: float,
) -> SchemeStageIncrementStencils:
    """Derive ESDIRK known-shift and error stencils from a tableau.

    The returned coefficients are suitable for the solved stage increments used
    by the built-in sequential ESDIRK schemes, not for raw derivative rates.
    """

    stage_count = len(tableau.b)
    increment_rows = [[0.0 for _ in range(stage_count)] for _ in range(stage_count)]
    increment_rows[0][0] = gamma

    for row_index in range(1, stage_count):
        for column_index, coefficient in enumerate(tableau.a[row_index]):
            increment_rows[row_index][column_index] = coefficient

        if increment_rows[row_index][row_index] != gamma:
            raise ValueError("ESDIRK stage rows must share the supplied gamma.")

    rate_weights_from_increments: list[tuple[float, ...]] = []
    for row_index, row in enumerate(increment_rows):
        diagonal = row[row_index]
        if diagonal == 0.0:
            raise ValueError("stage-increment basis must be lower triangular.")

        weights = [0.0 for _ in range(stage_count)]
        weights[row_index] = 1.0 / diagonal

        for previous_index in range(row_index):
            previous_scale = row[previous_index] / diagonal
            previous_weights = rate_weights_from_increments[previous_index]
            for column_index, coefficient in enumerate(previous_weights):
                weights[column_index] -= previous_scale * coefficient

        rate_weights_from_increments.append(tuple(weights))

    def convert_rate_weights(rate_weights: Iterable[float]) -> tuple[float, ...]:
        converted = [0.0 for _ in range(stage_count)]
        for rate_index, rate_weight in enumerate(rate_weights):
            basis_weights = rate_weights_from_increments[rate_index]
            for column_index, coefficient in enumerate(basis_weights):
                converted[column_index] += rate_weight * coefficient
        return tuple(converted)

    known_shifts = []
    for row_index, row in enumerate(tableau.a):
        if row_index == 0:
            known_shifts.append(())
            continue

        known_shifts.append(convert_rate_weights(row[:row_index])[:row_index])

    high_delta = convert_rate_weights(tableau.b)
    low_delta = (
        convert_rate_weights(tableau.b_embedded)
        if tableau.b_embedded is not None
        else ()
    )
    error_delta = (
        tuple(
            high - low
            for high, low in zip(high_delta, low_delta, strict=True)
        )
        if low_delta
        else ()
    )

    return SchemeStageIncrementStencils(
        known_shifts=tuple(known_shifts),
        high_delta=high_delta,
        low_delta=low_delta,
        error_delta=error_delta,
    )


__all__ = [
    "SchemeStencil",
    "SchemeStencilCoefficient",
    "SchemeStageIncrementStencils",
    "SchemeStencilTableau",
    "esdirk_stage_increment_stencils",
]
