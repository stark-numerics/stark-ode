from __future__ import annotations

"""
Dense support workers for Krylov subspace inverters.

These classes hold the small pieces of linear algebra that sit inside the
larger block-valued inverter algorithms:

- `Arnoldi` builds an orthonormal Krylov basis
- `GivensRotations` maintains the QR-like updates used by GMRES
- `HessenbergLeastSquares` stores and solves the tiny least-squares problem

They are separated from the inverter classes so the higher-level algorithms can
read like the iterative methods they implement rather than a tangle of dense
bookkeeping.
"""

import numpy as np

from stark.accelerators import AcceleratorAbsent
from stark.contracts import AcceleratorLike, Block
from stark.block.operator import BlockOperator
from stark.inverters.support.workspace import InverterWorkspace

def _givens(upper: float, lower: float) -> tuple[float, float]:
    """Return the cosine and sine of the Givens rotation that zeroes `lower`."""
    if lower == 0.0:
        return 1.0, 0.0
    radius = (upper * upper + lower * lower) ** 0.5
    return upper / radius, lower / radius


def _apply_previous_rotations(hessenberg, cosines, sines, column: int) -> None:
    """Apply all earlier Givens rotations to one Hessenberg column."""
    for row in range(column):
        rotated_upper = cosines[row] * hessenberg[row, column] + sines[row] * hessenberg[row + 1, column]
        rotated_lower = -sines[row] * hessenberg[row, column] + cosines[row] * hessenberg[row + 1, column]
        hessenberg[row, column] = rotated_upper
        hessenberg[row + 1, column] = rotated_lower


def _apply_new_rotation(hessenberg, residual_vector, cosines, sines, column: int) -> float:
    """Create and apply the next Givens rotation, returning the new residual estimate."""
    cosine, sine = _givens(hessenberg[column, column], hessenberg[column + 1, column])
    cosines[column] = cosine
    sines[column] = sine
    hessenberg[column, column] = cosine * hessenberg[column, column] + sine * hessenberg[column + 1, column]
    hessenberg[column + 1, column] = 0.0

    next_residual = -sine * residual_vector[column]
    residual_vector[column] = cosine * residual_vector[column]
    residual_vector[column + 1] = next_residual
    return abs(next_residual)


def _back_substitute(hessenberg, residual_vector, coefficients, size: int) -> None:
    """Solve the upper-triangular least-squares system produced by GMRES."""
    for row in range(size - 1, -1, -1):
        total = residual_vector[row]
        for column in range(row + 1, size):
            total -= hessenberg[row, column] * coefficients[column]
        coefficients[row] = total / hessenberg[row, row]


class GivensRotations:
    """
    Cached Givens rotations for one restarted Krylov window.

    GMRES applies a sequence of plane rotations to the Hessenberg matrix so the
    residual norm can be updated cheaply after each new Krylov column. This
    worker owns the rotation coefficients and the tiny dense kernels that apply
    them.
    """

    __slots__ = ("cosines", "sines", "_apply_previous", "_apply_new")

    def __init__(self, restart: int, accelerator: AcceleratorLike | None = None) -> None:
        self.cosines = np.zeros(restart, dtype=np.float64)
        self.sines = np.zeros(restart, dtype=np.float64)
        resolved_accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self._apply_previous = resolved_accelerator.decorate(_apply_previous_rotations)
        self._apply_new = _apply_new_rotation
        hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
        residual_vector = np.zeros(restart + 1, dtype=np.float64)
        resolved_accelerator.compile_examples(_givens, (1.0, 0.5))
        resolved_accelerator.compile_examples(self._apply_previous, (hessenberg, self.cosines, self.sines, 0))

    def reset(self) -> None:
        """Clear all stored rotations before a new restart window begins."""
        self.cosines.fill(0.0)
        self.sines.fill(0.0)

    def apply_previous(self, hessenberg, column: int) -> None:
        """Apply every previously constructed rotation to the current column."""
        self._apply_previous(hessenberg, self.cosines, self.sines, column)

    def apply_new(self, hessenberg, residual_vector, column: int) -> float:
        """Construct the next rotation, apply it, and return the residual estimate."""
        return self._apply_new(hessenberg, residual_vector, self.cosines, self.sines, column)


class HessenbergLeastSquares:
    """
    Dense least-squares data for one restarted Krylov window.

    In GMRES and FGMRES the large operator problem is reduced to a small dense
    least-squares problem over the Krylov basis. This worker stores that dense
    Hessenberg system and solves it once the current window ends.
    """

    __slots__ = ("restart", "hessenberg", "residual_vector", "coefficients", "_back_substitute")

    def __init__(self, restart: int, accelerator: AcceleratorLike | None = None) -> None:
        self.restart = restart
        self.hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
        self.residual_vector = np.zeros(restart + 1, dtype=np.float64)
        self.coefficients = np.zeros(restart, dtype=np.float64)
        resolved_accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self._back_substitute = resolved_accelerator.decorate(_back_substitute)
        resolved_accelerator.compile_examples(
            self._back_substitute,
            (self.hessenberg, self.residual_vector, self.coefficients, 1),
        )

    def reset(self, beta: float, window: int) -> None:
        """Initialize a fresh least-squares problem with leading residual `beta`."""
        self.hessenberg.fill(0.0)
        self.residual_vector.fill(0.0)
        self.coefficients.fill(0.0)
        self.residual_vector[0] = beta
        if window < self.restart:
            self.hessenberg[:, window:] = 0.0

    def solve(self, width: int) -> np.ndarray:
        """Solve the triangularized least-squares problem for `width` basis vectors."""
        self._back_substitute(self.hessenberg, self.residual_vector, self.coefficients, width)
        return self.coefficients


class Arnoldi:
    """
    Reusable Arnoldi worker for block-valued Krylov methods.

    Arnoldi orthonormalizes the sequence

        v, A v, A^2 v, ...

    into a basis `q_0, q_1, ...` and simultaneously builds the small Hessenberg
    matrix seen by GMRES-family methods. In STARK the basis vectors are `Block`
    objects, and the operator action is provided by `BlockOperator`.
    """

    __slots__ = ("workspace", "restart", "size", "basis", "work", "temporary")

    def __init__(self, workspace: InverterWorkspace, restart: int) -> None:
        self.workspace = workspace
        self.restart = restart
        self.size = -1
        self.basis = []
        self.work = None
        self.temporary = None

    def ensure_size(self, size: int) -> None:
        """Allocate or resize the cached basis and scratch blocks."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.work = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)
        self.basis = [workspace.allocate_block(size) for _ in range(self.restart + 1)]

    def start(self, residual: Block, beta: float) -> None:
        """Normalize the initial residual to produce the first Krylov basis vector."""
        self.workspace.scale_block(self.basis[0], 1.0 / beta, residual)

    def build_column(
        self,
        column: int,
        operator: BlockOperator,
        search_vector: Block,
        least_squares: HessenbergLeastSquares,
        rotations: GivensRotations,
    ) -> None:
        """
        Build one Arnoldi column and rotate it into the current least-squares form.

        The steps are the textbook Arnoldi procedure:

        1. apply the operator to the search vector
        2. orthogonalize against the existing basis
        3. normalize the remainder into the next basis vector
        4. apply earlier Givens rotations so the Hessenberg column stays aligned
           with the GMRES least-squares problem
        """
        workspace = self.workspace
        work = self.work
        temporary = self.temporary
        assert work is not None
        assert temporary is not None
        hessenberg = least_squares.hessenberg

        operator(search_vector, work)
        for row in range(column + 1):
            hessenberg[row, column] = workspace.inner_product(work, self.basis[row])
            workspace.combine2_block(temporary, 1.0, work, -hessenberg[row, column], self.basis[row])
            workspace.copy_block(work, temporary)

        hessenberg[column + 1, column] = workspace.norm(work)
        if hessenberg[column + 1, column] > 0.0:
            workspace.scale_block(self.basis[column + 1], 1.0 / hessenberg[column + 1, column], work)

        rotations.apply_previous(hessenberg, column)


__all__ = [
    "Arnoldi",
    "GivensRotations",
    "HessenbergLeastSquares",
]










