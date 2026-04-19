from __future__ import annotations

"""
Flexible restarted GMRES for STARK block-valued linear systems.

FGMRES differs from ordinary GMRES by allowing the preconditioning operation to
change from one Krylov column to the next. This is the variant described by
Saad in:

    Y. Saad,
    "A Flexible Inner-Outer Preconditioned GMRES Algorithm",
    SIAM Journal on Scientific Computing 14(2), 1993.

In STARK terms, the "preconditioner" is just another configured worker that
acts on `Block` objects. When no preconditioner is supplied, FGMRES reduces to
a slightly more general but somewhat more expensive GMRES-like scheme.
"""

from stark.contracts import AcceleratorLike, Block, InnerProduct, PreconditionerLike, Workbench
from stark.block.operator import BlockOperator
from stark.execution.safety import Safety
from stark.inverters.base import InverterBaseRestartedKrylov
from stark.inverters.descriptor import InverterDescriptor
from stark.inverters.policy import InverterPolicy
from stark.machinery.linear_algebra.krylov import Arnoldi, GivensRotations, HessenbergLeastSquares
from stark.inverters.tolerance import InverterTolerance
from stark.execution.tolerance import Tolerance


class FGMRESCycle:
    """
    One restarted FGMRES window with cached search and basis blocks.

    Unlike plain GMRES, the vectors used to *generate* the Krylov basis and the
    vectors used to *assemble the final correction* are not necessarily the
    same. FGMRES therefore stores a second basis, `search_basis`, containing
    the preconditioned directions.
    """

    __slots__ = (
        "workspace",
        "restart",
        "size",
        "applied",
        "residual",
        "correction",
        "arnoldi",
        "search_basis",
        "rotations",
        "least_squares",
    )

    def __init__(self, workspace: InverterWorkspace, restart: int, accelerator: AcceleratorLike | None = None) -> None:
        self.workspace = workspace
        self.restart = restart
        self.size = -1
        self.applied = None
        self.residual = None
        self.correction = None
        self.search_basis = []
        self.arnoldi = Arnoldi(workspace, restart)
        self.rotations = GivensRotations(restart, accelerator=accelerator)
        self.least_squares = HessenbergLeastSquares(restart, accelerator=accelerator)

    def ensure_size(self, size: int) -> None:
        """Allocate or resize all cached blocks for one FGMRES window."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.applied = workspace.allocate_block(size)
        self.residual = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
        self.search_basis = [workspace.allocate_block(size) for _ in range(self.restart)]
        self.arnoldi.ensure_size(size)

    def initial_residual(self, out: Block, rhs: Block, operator: BlockOperator) -> float:
        """Compute `r = rhs - A out` in cached storage and return its norm."""
        workspace = self.workspace
        applied = self.applied
        residual = self.residual
        assert applied is not None
        assert residual is not None
        operator(applied, out)
        workspace.combine2_block(residual, 1.0, rhs, -1.0, applied)
        return workspace.norm(residual)

    def run(
        self,
        out: Block,
        rhs: Block,
        operator: BlockOperator,
        tolerance: InverterTolerance,
        policy: InverterPolicy,
        rhs_norm: float,
        remaining_iterations: int,
        apply_preconditioner,
    ) -> tuple[int, float]:
        """
        Run one restarted FGMRES window and update `out` in place.

        `apply_preconditioner` is written as a worker call so the outer inverter
        can inject either a real right preconditioner or a simple copy path.
        """
        workspace = self.workspace
        residual = self.residual
        assert residual is not None
        beta = workspace.norm(residual)
        window = min(self.restart, remaining_iterations)
        self.rotations.reset()
        self.least_squares.reset(beta, window)
        self.arnoldi.start(residual, beta)

        last_column = -1
        for column in range(window):
            apply_preconditioner(self.search_basis[column], self.arnoldi.basis[column])
            self.arnoldi.build_column(
                column,
                operator,
                self.search_basis[column],
                self.least_squares,
                self.rotations,
            )
            residual_estimate = self.rotations.apply_new(
                self.least_squares.hessenberg,
                self.least_squares.residual_vector,
                column,
            )
            last_column = column
            if tolerance.accepts(residual_estimate, rhs_norm):
                self._apply_correction(out, last_column + 1)
                return column + 1, self.initial_residual(out, rhs, operator)

        self._apply_correction(out, last_column + 1)
        return window, self.initial_residual(out, rhs, operator)

    def _apply_correction(self, out: Block, width: int) -> None:
        """Form the correction from the stored preconditioned search basis."""
        if width <= 0:
            return

        workspace = self.workspace
        correction = self.correction
        temporary = self.arnoldi.temporary
        assert correction is not None
        assert temporary is not None
        workspace.zero_block(correction)
        coefficients = self.least_squares.solve(width)

        for index in range(width):
            workspace.combine2_block(temporary, 1.0, correction, coefficients[index], self.search_basis[index])
            workspace.copy_block(correction, temporary)

        workspace.combine2_block(temporary, 1.0, out, 1.0, correction)
        workspace.copy_block(out, temporary)


class InverterFGMRES(InverterBaseRestartedKrylov):
    """
    Restarted flexible GMRES with an optional right preconditioner.

    This inverter solves `A x = b` over STARK blocks in the same general way as
    `InverterGMRES`, but it allows the preconditioning action to vary across the
    Krylov window. That makes it a good fit when the "preconditioner" is itself
    an iterative or stateful worker.

    In the current library the preconditioner slot accepts any `PreconditionerLike`
    object. If no preconditioner is provided, FGMRES falls back to the identity
    action and behaves like an unpreconditioned flexible method.

    Reference:
        Saad (1993), SIAM J. Sci. Comput. 14(2).
    """

    __slots__ = ()

    descriptor = InverterDescriptor("FGMRES", "Flexible Restarted GMRES")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: PreconditionerLike | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        self.initialise_inverter(
            workbench,
            inner_product,
            tolerance=tolerance,
            policy=policy,
            preconditioner=preconditioner,
            safety=safety,
            accelerator=accelerator,
        )

    def make_cycle(self) -> FGMRESCycle:
        return FGMRESCycle(self.workspace, self.policy.restart, accelerator=self.accelerator)

__all__ = ["InverterFGMRES"]











