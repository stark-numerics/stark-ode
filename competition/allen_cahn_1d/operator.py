from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class AllenCahnJacobianPeriodicTridiagonal:
    """Matrix-free periodic tridiagonal Jacobian action for Allen-Cahn 1D."""

    state_u: Any
    diffusivity: float
    inv_dx2: float
    off: float = field(init=False)
    center: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.off = self.diffusivity * self.inv_dx2
        self.center = (1.0 - 3.0 * self.state_u * self.state_u - 2.0 * self.off).astype(np.float64)

    def __call__(self, source: Any, target: Any) -> None:
        x = source.du
        y = target.du
        off = self.off
        center = self.center
        if x.size == 1:
            y[0] = (center[0] + 2.0 * off) * x[0]
            return
        y[1:-1] = off * x[:-2] + center[1:-1] * x[1:-1] + off * x[2:]
        y[0] = off * x[-1] + center[0] * x[0] + off * x[1]
        y[-1] = off * x[-2] + center[-1] * x[-1] + off * x[0]


@dataclass(slots=True)
class AllenCahnPreconditionerPeriodicTridiagonal:
    """Exact cyclic-tridiagonal preconditioner for Allen-Cahn Newton systems."""

    def __call__(self, operator: Any, source: Any, target: Any) -> None:
        action = self._action(operator)
        if action is None:
            target[0].du[:] = source[0].du
            return

        differential, jacobian = action
        alpha = differential.alpha
        off = -alpha * jacobian.off
        diagonal = 1.0 - alpha * jacobian.center
        target[0].du[:] = _solve_cyclic_tridiagonal(
            lower=off,
            diagonal=diagonal,
            upper=off,
            corner_upper=off,
            corner_lower=off,
            rhs=source[0].du,
        )

    @staticmethod
    def _action(operator: Any):
        try:
            if len(operator) != 1:
                return None
            differential = operator[0]
        except Exception:
            return None

        jacobian_operator = getattr(differential, "jacobian", None)
        jacobian = getattr(jacobian_operator, "apply", None)
        if isinstance(jacobian, AllenCahnJacobianPeriodicTridiagonal):
            return differential, jacobian
        return None


def _solve_tridiagonal(lower: float, diagonal: np.ndarray, upper: float, rhs: np.ndarray) -> np.ndarray:
    size = rhs.size
    c_prime = np.empty(size - 1, dtype=np.float64)
    d_prime = np.empty(size, dtype=np.float64)

    pivot = diagonal[0]
    c_prime[0] = upper / pivot
    d_prime[0] = rhs[0] / pivot

    for index in range(1, size - 1):
        pivot = diagonal[index] - lower * c_prime[index - 1]
        c_prime[index] = upper / pivot
        d_prime[index] = (rhs[index] - lower * d_prime[index - 1]) / pivot

    pivot = diagonal[size - 1] - lower * c_prime[size - 2]
    d_prime[size - 1] = (rhs[size - 1] - lower * d_prime[size - 2]) / pivot

    solution = np.empty(size, dtype=np.float64)
    solution[size - 1] = d_prime[size - 1]
    for index in range(size - 2, -1, -1):
        solution[index] = d_prime[index] - c_prime[index] * solution[index + 1]
    return solution


def _solve_cyclic_tridiagonal(
    *,
    lower: float,
    diagonal: np.ndarray,
    upper: float,
    corner_upper: float,
    corner_lower: float,
    rhs: np.ndarray,
) -> np.ndarray:
    size = rhs.size
    if size == 1:
        return rhs / (diagonal[0] + corner_upper + corner_lower)
    if size == 2:
        matrix = np.array(
            [
                [diagonal[0], upper + corner_upper],
                [lower + corner_lower, diagonal[1]],
            ],
            dtype=np.float64,
        )
        return np.linalg.solve(matrix, rhs)

    gamma = -diagonal[0]
    adjusted = diagonal.copy()
    adjusted[0] = diagonal[0] - gamma
    adjusted[-1] = diagonal[-1] - corner_upper * corner_lower / gamma

    solution = _solve_tridiagonal(lower, adjusted, upper, rhs)
    update = np.zeros(size, dtype=np.float64)
    update[0] = gamma
    update[-1] = corner_upper
    z = _solve_tridiagonal(lower, adjusted, upper, update)

    denominator = 1.0 + z[0] + corner_lower * z[-1] / gamma
    factor = (solution[0] + corner_lower * solution[-1] / gamma) / denominator
    return solution - factor * z


__all__ = [
    "AllenCahnJacobianPeriodicTridiagonal",
    "AllenCahnPreconditionerPeriodicTridiagonal",
]
