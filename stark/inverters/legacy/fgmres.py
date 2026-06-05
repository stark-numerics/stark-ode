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

from stark.block import Block
from stark.contracts import Accelerator, InnerProduct, LegacyInverterPreconditionerLike, Allocator
from stark.block.operator import BlockOperatorDiagonal
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
from stark.core import Tolerance
from stark.inverters.configuration import InverterConfiguration


# Optional extension: adds human-readable inverter metadata and formatting helpers.
# Provides: short_name, full_name, __repr__, and __str__.
@with_inverter_display_methods
@with_inverter_binding_methods
class InverterFGMRES:
    """
    Restarted flexible GMRES with an optional right preconditioner.

    This inverter solves `A x = b` over STARK blocks in the same general way as
    `InverterGMRES`, but it allows the preconditioning action to vary across the
    Krylov window. That makes it a good fit when the "preconditioner" is itself
    an iterative or stateful worker.

    In the current library the preconditioner slot accepts any `LegacyInverterPreconditionerLike`
    object. If no preconditioner is provided, FGMRES falls back to the identity
    action and behaves like an unpreconditioned flexible method.

    Reference:
        Saad (1993), SIAM J. Sci. Comput. 14(2).
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

    descriptor = InverterDescriptor("FGMRES", "Flexible Restarted GMRES")

    def __init__(
        self,
        allocator: Allocator,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        configuration: InverterConfiguration | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: LegacyInverterPreconditionerLike | None = None,
        safety: InverterSafety | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        # Installs self.workspace; see stark.inverters.legacy_support.workspace for its operations.
        initialise_inverter_runtime(
            self,
            allocator,
            inner_product,
            tolerance=tolerance,
            configuration=configuration,
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
        """Allocate or resize the cached blocks for one FGMRES restart window."""
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
            raise RuntimeError("FGMRES inverter must be bound to an operator before use.")

        inverter_tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        rhs_norm = workspace.norm(rhs)
        residual_norm = self.initial_residual(rhs, out, operator)
        self._monitor_initial_residual = residual_norm
        self._monitor_final_residual = residual_norm
        self._monitor_iteration_count = 0
        if inverter_tolerance.accepts(residual_norm, rhs_norm):
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
            if inverter_tolerance.accepts(residual_norm, rhs_norm):
                return

        raise RuntimeError(
            "FGMRES failed to converge within "
            f"{policy.max_iterations} iterations (residual={residual_norm:g})."
        )

    def initial_residual(self, rhs: Block, out: Block, operator: BlockOperatorDiagonal) -> float:
        """Compute `r = rhs - A out` in cached storage and return its norm."""
        workspace = self.workspace
        operator(out, self.applied)
        workspace.combine2_block(1.0, rhs, -1.0, self.applied, self.residual)
        return workspace.norm(self.residual)

    def run_window(
        self,
        rhs: Block,
        out: Block,
        operator: BlockOperatorDiagonal,
        rhs_norm: float,
        remaining_iterations: int,
    ) -> tuple[int, float]:
        """
        Run one restarted FGMRES window and update `out` in place.

        The preconditioner may be stateful or iterative, so the correction is
        assembled from the stored preconditioned search directions.
        """
        inverter_tolerance = self.tolerance
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
            if inverter_tolerance.accepts(residual_estimate, rhs_norm):
                self.apply_correction(out, last_column + 1)
                return column + 1, self.initial_residual(rhs, out, operator)

        self.apply_correction(out, last_column + 1)
        return window, self.initial_residual(rhs, out, operator)

    def apply_correction(self, out: Block, width: int) -> None:
        """Form the correction from the stored preconditioned search basis."""
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


__all__ = ["InverterFGMRES"]










