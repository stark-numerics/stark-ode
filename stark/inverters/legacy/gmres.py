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
the Krylov support workers in `stark.inverters.legacy_support.krylov`.
"""

from stark.block import Block
from stark.contracts import AcceleratorLike, InnerProduct, InverterPreconditionerLike, Allocator
from stark.block.operator import BlockOperator
from stark.inverters.legacy_support.descriptor import InverterDescriptor
from stark.inverters.legacy_support.policy import InverterPolicy
from stark.inverters.legacy_support.safety import InverterSafety
from stark.inverters.legacy_support import (
    initialise_inverter_runtime,
    validate_restarted_inverter_policy,
    with_inverter_binding_methods,
    with_inverter_display_methods,
)
from stark.inverters.legacy_support.krylov import Arnoldi, GivensRotations, HessenbergLeastSquares
from stark.executor.tolerance import ExecutorTolerance


@with_inverter_display_methods
@with_inverter_binding_methods
class InverterGMRES:
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

    __slots__ = (
        "accelerator",
        "applied",
        "arnoldi",
        "_bound_call",
        "_monitor",
        "_monitor_final_residual",
        "_monitor_initial_residual",
        "_monitor_iteration_count",
        "correction",
        "least_squares",
        "operator",
        "policy",
        "preconditioner",
        "redirect_call",
        "residual",
        "restart",
        "rotations",
        "safety",
        "search_basis",
        "size",
        "temporary",
        "tolerance",
        "workspace",
    )

    descriptor = InverterDescriptor("GMRES", "Restarted GMRES")

    def __init__(
        self,
        allocator: Allocator,
        inner_product: InnerProduct,
        ExecutorTolerance: ExecutorTolerance | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: InverterPreconditionerLike | None = None,
        safety: InverterSafety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        # Installs self.workspace; see stark.inverters.legacy_support.workspace for its operations.
        initialise_inverter_runtime(
            self,
            allocator,
            inner_product,
            tolerance=ExecutorTolerance,
            policy=policy,
            preconditioner=preconditioner,
            safety=safety,
            accelerator=accelerator,
        )
        validate_restarted_inverter_policy(self.policy)
        self.restart = self.policy.restart
        self.size = -1
        self.applied = self.workspace.allocate_block(0)
        self.residual = self.workspace.allocate_block(0)
        self.correction = self.workspace.allocate_block(0)
        self.temporary = self.workspace.allocate_block(0)
        self.search_basis = []
        self.arnoldi = Arnoldi(self.workspace, self.restart)
        self.rotations = GivensRotations(self.restart, accelerator=self.accelerator)
        self.least_squares = HessenbergLeastSquares(self.restart, accelerator=self.accelerator)

    def ensure_size(self, size: int) -> None:
        """Allocate or resize the cached blocks for one GMRES restart window."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.applied = workspace.allocate_block(size)
        self.residual = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)
        self.search_basis = [workspace.allocate_block(size) for _ in range(self.restart)]
        self.arnoldi.ensure_size(size)

    def solve_prepared(self, rhs: Block, out: Block) -> None:
        operator = self.operator
        if operator is None:
            raise RuntimeError("GMRES inverter must be bound to an operator before use.")

        ExecutorTolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        rhs_norm = workspace.norm(rhs)
        residual_norm = self.initial_residual(rhs, out, operator)
        self._monitor_initial_residual = residual_norm
        self._monitor_final_residual = residual_norm
        self._monitor_iteration_count = 0
        if ExecutorTolerance.accepts(residual_norm, rhs_norm):
            return

        iterations = 0
        while iterations < policy.max_iterations:
            used_iterations, residual_norm = self.run_window(
                rhs,
                out,
                operator,
                rhs_norm,
                policy.max_iterations - iterations,
            )
            iterations += used_iterations
            self._monitor_iteration_count = iterations
            self._monitor_final_residual = residual_norm
            if ExecutorTolerance.accepts(residual_norm, rhs_norm):
                return

        raise RuntimeError(
            "GMRES failed to converge within "
            f"{policy.max_iterations} iterations (residual={residual_norm:g})."
        )

    def initial_residual(self, rhs: Block, out: Block, operator: BlockOperator) -> float:
        """
        Compute `r = rhs - A out` in cached storage and return its norm.

        GMRES treats `out` as the current iterate and improves it in place.
        """
        workspace = self.workspace
        operator(out, self.applied)
        workspace.combine2_block(1.0, rhs, -1.0, self.applied, self.residual)
        return workspace.norm(self.residual)

    def run_window(
        self,
        rhs: Block,
        out: Block,
        operator: BlockOperator,
        rhs_norm: float,
        remaining_iterations: int,
    ) -> tuple[int, float]:
        """
        Run one restarted GMRES window and update `out` in place.

        Returns the number of Krylov iterations used in this window and the true
        residual norm after the correction has been applied.
        """
        ExecutorTolerance = self.tolerance
        beta = self.workspace.norm(self.residual)
        window = min(self.restart, remaining_iterations)
        self.rotations.reset()
        self.least_squares.reset(beta, window)
        self.arnoldi.start(self.residual, beta)

        last_column = -1
        for column in range(window):
            self.preconditioner(self.arnoldi.basis[column], self.search_basis[column])
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
            if ExecutorTolerance.accepts(residual_estimate, rhs_norm):
                self.apply_correction(out, last_column + 1)
                return column + 1, self.initial_residual(rhs, out, operator)

        self.apply_correction(out, last_column + 1)
        return window, self.initial_residual(rhs, out, operator)

    def apply_correction(self, out: Block, width: int) -> None:
        """Form the Krylov correction and accumulate it into `out`."""
        if width <= 0:
            return

        workspace = self.workspace
        workspace.zero_block(self.correction)
        coefficients = self.least_squares.solve(width)

        for index in range(width):
            workspace.combine2_block(
                1.0,
                self.correction,
                coefficients[index],
                self.search_basis[index],
                self.temporary,
            )
            workspace.copy_block(self.correction, self.temporary)

        workspace.combine2_block(1.0, out, 1.0, self.correction, self.temporary)
        workspace.copy_block(out, self.temporary)


__all__ = ["InverterGMRES"]










