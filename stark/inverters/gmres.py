from __future__ import annotations

"""
Restarted GMRES for STARK block-valued linear systems.

GMRES builds a Krylov basis

    span{r0, A r0, A^2 r0, ...}

and chooses the correction whose residual is minimal over that basis. In the
standard matrix setting this is the method introduced by Saad and Schultz:

    Y. Saad and M. H. Schultz,
    "GMRES: A Generalized Minimal Residual Algorithm for Solving
    Nonsymmetric Linear Systems",
    SIAM Journal on Scientific and Statistical Computing 7(3), 1986.

STARK uses the same algorithmic structure, but replaces vectors by `Block`
objects and matrix-vector products by `BlockOperator` applications. The dense
small linear algebra that lives inside one GMRES restart window is handled by
the Krylov support workers in `stark.machinery.linear_algebra.krylov`.
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


class GMRESCycle:
    """
    One restarted GMRES window with all block storage allocated up front.

    The outer `InverterGMRES` owns convergence policy and restart bookkeeping.
    This worker owns the actual work of one Krylov cycle:

    1. form the initial residual
    2. build an Arnoldi basis
    3. update the Hessenberg least-squares problem with Givens rotations
    4. reconstruct the correction and apply it to the current iterate

    Keeping this in a dedicated worker keeps the top-level inverter readable and
    gives the hot path a stable set of scratch blocks.
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
        """Allocate or resize all cached blocks for a given block length."""
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
        """
        Compute `r = rhs - A out` in cached storage and return its norm.

        GMRES treats `out` as the current iterate and improves it in place.
        """
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
        Run one restarted GMRES window and update `out` in place.

        Returns the number of Krylov iterations used in this window and the true
        residual norm after the correction has been applied.
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
        """Form the Krylov correction and accumulate it into `out`."""
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


class InverterGMRES(InverterBaseRestartedKrylov):
    """
    Restarted GMRES on STARK blocks using a user-supplied inner product.

    Conceptually this is the standard restarted GMRES method for solving

        A x = b

    with the following STARK substitutions:

    - `x` and `b` are `Block` objects rather than flat arrays
    - `A` is supplied as a `BlockOperator`
    - all vector-space mechanics are delegated to `InverterWorkspace`

    A fixed right preconditioner can also be supplied. The preconditioner is
    treated as another configured worker that acts on `Block` objects.

    The inverter is configured once, bound to one operator, and then called
    repeatedly on preallocated blocks. This matches STARK's "configured worker"
    style and keeps the hot `__call__` path small.

    Reference:
        Saad and Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3).
    """

    __slots__ = ()

    descriptor = InverterDescriptor("GMRES", "Restarted GMRES")

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

    def make_cycle(self) -> GMRESCycle:
        return GMRESCycle(self.workspace, self.policy.restart, accelerator=self.accelerator)


__all__ = ["InverterGMRES"]











