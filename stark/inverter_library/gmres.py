from __future__ import annotations

import numpy as np

from stark.contracts import Block, InnerProduct, InverterLike, Workbench
from stark.inverter_support.block_operator import BlockOperator
from stark.inverter_support.descriptor import InverterDescriptor
from stark.inverter_support.inversion import Inversion
from stark.inverter_support.workspace import InverterWorkspace

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional accelerator
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def _optional_njit(function):
    return njit(cache=True)(function) if NUMBA_AVAILABLE else function


@_optional_njit
def _givens(upper: float, lower: float) -> tuple[float, float]:
    if lower == 0.0:
        return 1.0, 0.0
    radius = (upper * upper + lower * lower) ** 0.5
    return upper / radius, lower / radius


@_optional_njit
def _apply_previous_rotations(hessenberg, cosines, sines, column: int) -> None:
    for row in range(column):
        rotated_upper = cosines[row] * hessenberg[row, column] + sines[row] * hessenberg[row + 1, column]
        rotated_lower = -sines[row] * hessenberg[row, column] + cosines[row] * hessenberg[row + 1, column]
        hessenberg[row, column] = rotated_upper
        hessenberg[row + 1, column] = rotated_lower


@_optional_njit
def _apply_new_rotation(hessenberg, residual_vector, cosines, sines, column: int) -> float:
    cosine, sine = _givens(hessenberg[column, column], hessenberg[column + 1, column])
    cosines[column] = cosine
    sines[column] = sine
    hessenberg[column, column] = cosine * hessenberg[column, column] + sine * hessenberg[column + 1, column]
    hessenberg[column + 1, column] = 0.0

    next_residual = -sine * residual_vector[column]
    residual_vector[column] = cosine * residual_vector[column]
    residual_vector[column + 1] = next_residual
    return abs(next_residual)


@_optional_njit
def _back_substitute(hessenberg, residual_vector, coefficients, size: int) -> None:
    for row in range(size - 1, -1, -1):
        total = residual_vector[row]
        for column in range(row + 1, size):
            total -= hessenberg[row, column] * coefficients[column]
        coefficients[row] = total / hessenberg[row, row]


class _GMRESCycle:
    """One reusable restarted GMRES cycle with cached storage."""

    __slots__ = (
        "workspace",
        "restart",
        "size",
        "applied",
        "residual",
        "work",
        "temporary",
        "correction",
        "basis",
        "hessenberg",
        "cosines",
        "sines",
        "residual_vector",
        "coefficients",
    )

    def __init__(self, workspace: InverterWorkspace, restart: int) -> None:
        self.workspace = workspace
        self.restart = restart
        self.size = -1
        self.applied = None
        self.residual = None
        self.work = None
        self.temporary = None
        self.correction = None
        self.basis = []
        self.hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
        self.cosines = np.zeros(restart, dtype=np.float64)
        self.sines = np.zeros(restart, dtype=np.float64)
        self.residual_vector = np.zeros(restart + 1, dtype=np.float64)
        self.coefficients = np.zeros(restart, dtype=np.float64)

    def ensure_size(self, size: int) -> None:
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.applied = workspace.allocate_block(size)
        self.residual = workspace.allocate_block(size)
        self.work = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
        self.basis = [workspace.allocate_block(size) for _ in range(self.restart + 1)]

    def initial_residual(self, out: Block, rhs: Block, operator: BlockOperator) -> float:
        workspace = self.workspace
        applied = self._require_applied()
        residual = self._require_residual()
        operator(applied, out)
        workspace.combine2_block(residual, 1.0, rhs, -1.0, applied)
        return workspace.norm(residual)

    def run(self, out: Block, rhs: Block, operator: BlockOperator, rhs_norm: float, inversion: Inversion, remaining_iterations: int) -> tuple[int, float]:
        workspace = self.workspace
        beta = workspace.norm(self._require_residual())
        window = min(self.restart, remaining_iterations)
        self._reset_cycle(beta, window)
        workspace.scale_block(self.basis[0], 1.0 / beta, self._require_residual())

        last_column = -1
        for column in range(window):
            self._build_arnoldi_column(column, operator)
            residual_estimate = _apply_new_rotation(
                self.hessenberg,
                self.residual_vector,
                self.cosines,
                self.sines,
                column,
            )
            last_column = column
            if inversion.accepts(residual_estimate, rhs_norm):
                self._apply_correction(out, last_column + 1)
                return column + 1, self.initial_residual(out, rhs, operator)

        self._apply_correction(out, last_column + 1)
        return window, self.initial_residual(out, rhs, operator)

    def _reset_cycle(self, beta: float, window: int) -> None:
        self.hessenberg.fill(0.0)
        self.cosines.fill(0.0)
        self.sines.fill(0.0)
        self.residual_vector.fill(0.0)
        self.coefficients.fill(0.0)
        self.residual_vector[0] = beta
        if window < self.restart:
            self.hessenberg[:, window:] = 0.0

    def _build_arnoldi_column(self, column: int, operator: BlockOperator) -> None:
        workspace = self.workspace
        work = self._require_work()
        temporary = self._require_temporary()
        operator(work, self.basis[column])

        for row in range(column + 1):
            self.hessenberg[row, column] = workspace.inner_product(work, self.basis[row])
            workspace.combine2_block(temporary, 1.0, work, -self.hessenberg[row, column], self.basis[row])
            workspace.copy_block(work, temporary)

        self.hessenberg[column + 1, column] = workspace.norm(work)
        if self.hessenberg[column + 1, column] > 0.0:
            workspace.scale_block(self.basis[column + 1], 1.0 / self.hessenberg[column + 1, column], work)

        _apply_previous_rotations(self.hessenberg, self.cosines, self.sines, column)

    def _apply_correction(self, out: Block, width: int) -> None:
        if width <= 0:
            return

        workspace = self.workspace
        correction = self._require_correction()
        temporary = self._require_temporary()
        workspace.zero_block(correction)
        _back_substitute(self.hessenberg, self.residual_vector, self.coefficients, width)

        for index in range(width):
            workspace.combine2_block(temporary, 1.0, correction, self.coefficients[index], self.basis[index])
            workspace.copy_block(correction, temporary)

        workspace.combine2_block(temporary, 1.0, out, 1.0, correction)
        workspace.copy_block(out, temporary)

    def _require_applied(self) -> Block:
        assert self.applied is not None
        return self.applied

    def _require_residual(self) -> Block:
        assert self.residual is not None
        return self.residual

    def _require_work(self) -> Block:
        assert self.work is not None
        return self.work

    def _require_temporary(self) -> Block:
        assert self.temporary is not None
        return self.temporary

    def _require_correction(self) -> Block:
        assert self.correction is not None
        return self.correction


class InverterGMRES:
    """Restarted GMRES on STARK blocks using a user-supplied inner product."""

    __slots__ = ("inversion", "operator", "workspace", "cycle", "safe", "_call")

    descriptor = InverterDescriptor("GMRES", "Restarted GMRES")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        inversion: Inversion | None = None,
        safe: bool = True,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.inversion = inversion if inversion is not None else Inversion()
        if self.inversion.max_iterations < 1:
            raise ValueError("Inversion.max_iterations must be at least 1.")
        if self.inversion.restart < 1:
            raise ValueError("Inversion.restart must be at least 1.")
        self.workspace = InverterWorkspace(workbench, translation_probe, inner_product)
        self.operator: BlockOperator | None = None
        self.cycle = _GMRESCycle(self.workspace, self.inversion.restart)
        self.safe = safe
        self._call = self._call_unbound

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"InverterGMRES(inversion={self.inversion!r}, safe={self.safe!r})"

    def __str__(self) -> str:
        return f"{self.short_name} with {self.inversion}"

    def bind(self, operator: BlockOperator) -> None:
        self.operator = operator
        self._call = self._call_safe if self.safe else self._call_unsafe

    def prepare(self, size: int) -> None:
        self.cycle.ensure_size(size)

    def __call__(self, out: Block, rhs: Block) -> None:
        self._call(out, rhs)

    def _call_unbound(self, out: Block, rhs: Block) -> None:
        del out, rhs
        raise RuntimeError("GMRES inverter must be bound to an operator before use.")

    def _call_safe(self, out: Block, rhs: Block) -> None:
        workspace = self.workspace
        workspace._check_size(out, rhs)
        self.prepare(len(out))
        self._solve_prepared(out, rhs)

    def _call_unsafe(self, out: Block, rhs: Block) -> None:
        self._solve_prepared(out, rhs)

    def _solve_prepared(self, out: Block, rhs: Block) -> None:
        operator = self.operator
        assert operator is not None
        inversion = self.inversion
        workspace = self.workspace
        rhs_norm = workspace.norm(rhs)
        beta = self.cycle.initial_residual(out, rhs, operator)
        if inversion.accepts(beta, rhs_norm):
            return

        iterations = 0
        while iterations < inversion.max_iterations:
            used_iterations, beta = self.cycle.run(
                out,
                rhs,
                operator,
                rhs_norm,
                inversion,
                inversion.max_iterations - iterations,
            )
            iterations += used_iterations
            if inversion.accepts(beta, rhs_norm):
                return

        raise RuntimeError(
            f"{self.short_name} failed to converge within "
            f"{inversion.max_iterations} iterations (residual={beta:g})."
        )


__all__ = ["InverterGMRES", "NUMBA_AVAILABLE"]
