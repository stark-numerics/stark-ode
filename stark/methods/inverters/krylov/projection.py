from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt


def _givens(upper: float, lower: float) -> tuple[float, float]:
    """Return the stable Givens rotation for a two-entry column tail."""

    if lower == 0.0:
        return 1.0, 0.0
    radius = sqrt(upper * upper + lower * lower)
    if radius == 0.0:
        return 1.0, 0.0
    return upper / radius, lower / radius


@dataclass(slots=True)
class InverterKrylovProjection:
    """Small Hessenberg least-squares projection for one Arnoldi window."""

    restart: int
    hessenberg: list[list[float]] = field(init=False)
    residual: list[float] = field(init=False)
    coefficients: list[float] = field(init=False)
    cosines: list[float] = field(init=False)
    sines: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.hessenberg = [
            [0.0 for _column in range(self.restart)]
            for _row in range(self.restart + 1)
        ]
        self.residual = [0.0 for _row in range(self.restart + 1)]
        self.coefficients = [0.0 for _column in range(self.restart)]
        self.cosines = [0.0 for _column in range(self.restart)]
        self.sines = [0.0 for _column in range(self.restart)]

    def reset(self, beta: float) -> None:
        for row in self.hessenberg:
            for column in range(len(row)):
                row[column] = 0.0
        for row in range(len(self.residual)):
            self.residual[row] = 0.0
        for column in range(len(self.coefficients)):
            self.coefficients[column] = 0.0
            self.cosines[column] = 0.0
            self.sines[column] = 0.0
        self.residual[0] = beta

    def apply_previous(self, column: int) -> None:
        """Apply rotations from earlier columns to a new Hessenberg column."""

        hessenberg = self.hessenberg
        for row in range(column):
            cosine = self.cosines[row]
            sine = self.sines[row]
            upper = hessenberg[row][column]
            lower = hessenberg[row + 1][column]
            hessenberg[row][column] = cosine * upper + sine * lower
            hessenberg[row + 1][column] = -sine * upper + cosine * lower

    def apply_new(self, column: int) -> float:
        """Triangularise the new column and return the residual estimate."""

        hessenberg = self.hessenberg
        cosine, sine = _givens(
            hessenberg[column][column],
            hessenberg[column + 1][column],
        )
        self.cosines[column] = cosine
        self.sines[column] = sine

        upper = hessenberg[column][column]
        lower = hessenberg[column + 1][column]
        hessenberg[column][column] = cosine * upper + sine * lower
        hessenberg[column + 1][column] = 0.0

        next_residual = -sine * self.residual[column]
        self.residual[column] = cosine * self.residual[column]
        self.residual[column + 1] = next_residual
        return abs(next_residual)

    def solve(self, width: int) -> list[float]:
        """Solve the current upper-triangular projected problem."""

        hessenberg = self.hessenberg
        residual = self.residual
        coefficients = self.coefficients
        for column in range(len(coefficients)):
            coefficients[column] = 0.0

        for row in range(width - 1, -1, -1):
            total = residual[row]
            for column in range(row + 1, width):
                total -= hessenberg[row][column] * coefficients[column]
            diagonal = hessenberg[row][row]
            coefficients[row] = 0.0 if diagonal == 0.0 else total / diagonal
        return coefficients


__all__ = ["InverterKrylovProjection"]
