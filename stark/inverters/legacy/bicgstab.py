from __future__ import annotations

"""
BiCGStab for STARK block-valued linear systems.

BiCGStab is a short-recurrence Krylov method derived from bi-conjugate
gradients, with an additional smoothing step intended to tame the erratic
residual behaviour of plain BiCG. The standard reference is:

    H. A. van der Vorst,
    "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG
    for the Solution of Nonsymmetric Linear Systems",
    SIAM Journal on Scientific and Statistical Computing 13(2), 1992.

Relative to GMRES-family methods, BiCGStab stores much less basis data, but
its convergence can be less predictable. That tradeoff makes it a useful
contrast inside the STARK inverter library.
"""

from stark.block import Block
from stark.contracts import AcceleratorLike, InnerProduct, LegacyInverterPreconditionerLike, Allocator
from stark.block.operator import BlockOperatorDiagonal
from stark.inverters.legacy_support.descriptor import InverterDescriptor
from stark.inverters.legacy_support.policy import InverterPolicy
from stark.inverters.legacy_support.safety import InverterSafety
from stark.inverters.legacy_support import (
    initialise_inverter_runtime,
    validate_inverter_policy,
    with_inverter_binding_methods,
    with_inverter_display_methods,
)
from stark.executor.tolerance import ExecutorTolerance


# Optional extension: adds human-readable inverter metadata and formatting helpers.
# Provides: short_name, full_name, __repr__, and __str__.
@with_inverter_display_methods
@with_inverter_binding_methods
class InverterBiCGStab:
    """
    Operator-based BiCGStab on STARK blocks.

    This inverter solves `A x = b` using a short-recurrence Krylov method that
    only needs a fixed amount of block storage, independent of the iteration
    count. That makes it cheaper in memory than GMRES-family methods, at the
    cost of somewhat rougher convergence behaviour.

    Reference:
        van der Vorst (1992), SIAM J. Sci. Stat. Comput. 13(2).
    """

    __slots__ = (
        "accelerator",
        "applied",
        "_bound_call",
        "_monitor",
        "_monitor_final_residual",
        "_monitor_initial_residual",
        "_monitor_iteration_count",
        "direction",
        "operator",
        "policy",
        "preconditioner",
        "preconditioned_direction",
        "preconditioned_s",
        "redirect_call",
        "residual",
        "s_buffer",
        "safety",
        "shadow",
        "size",
        "t_buffer",
        "temporary",
        "tolerance",
        "velocity",
        "workspace",
    )

    descriptor = InverterDescriptor("BiCGStab", "BiConjugate Gradient Stabilized")

    def __init__(
        self,
        allocator: Allocator,
        inner_product: InnerProduct,
        ExecutorTolerance: ExecutorTolerance | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: LegacyInverterPreconditionerLike | None = None,
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
        validate_inverter_policy(self.policy)
        self.size = -1
        self.applied = self.workspace.allocate_block(0)
        self.residual = self.workspace.allocate_block(0)
        self.shadow = self.workspace.allocate_block(0)
        self.direction = self.workspace.allocate_block(0)
        self.velocity = self.workspace.allocate_block(0)
        self.preconditioned_direction = self.workspace.allocate_block(0)
        self.preconditioned_s = self.workspace.allocate_block(0)
        self.s_buffer = self.workspace.allocate_block(0)
        self.t_buffer = self.workspace.allocate_block(0)
        self.temporary = self.workspace.allocate_block(0)

    def ensure_size(self, size: int) -> None:
        """Allocate or resize all cached blocks for one BiCGStab solve."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.applied = workspace.allocate_block(size)
        self.residual = workspace.allocate_block(size)
        self.shadow = workspace.allocate_block(size)
        self.direction = workspace.allocate_block(size)
        self.velocity = workspace.allocate_block(size)
        self.preconditioned_direction = workspace.allocate_block(size)
        self.preconditioned_s = workspace.allocate_block(size)
        self.s_buffer = workspace.allocate_block(size)
        self.t_buffer = workspace.allocate_block(size)
        self.temporary = workspace.allocate_block(size)

    def solve_prepared(self, rhs: Block, out: Block) -> None:
        operator = self.operator
        if operator is None:
            raise RuntimeError("BiCGStab inverter must be bound to an operator before use.")
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

        iterations, residual_norm = self.iterate(out, operator, rhs_norm)
        self._monitor_iteration_count = iterations
        self._monitor_final_residual = residual_norm
        if ExecutorTolerance.accepts(residual_norm, rhs_norm):
            return

        raise RuntimeError(
            "BiCGStab failed to converge within "
            f"{policy.max_iterations} iterations (residual={residual_norm:g})."
        )

    def initial_residual(self, rhs: Block, out: Block, operator: BlockOperatorDiagonal) -> float:
        """
        Compute `r = rhs - A out`, initialize the BiCG shadow state, and return
        the residual norm.
        """
        workspace = self.workspace
        operator(out, self.applied)
        workspace.combine2_block(1.0, rhs, -1.0, self.applied, self.residual)
        workspace.copy_block(self.shadow, self.residual)
        workspace.zero_block(self.direction)
        workspace.zero_block(self.velocity)
        return workspace.norm(self.residual)

    def iterate(self, out: Block, operator: BlockOperatorDiagonal, rhs_norm: float) -> tuple[int, float]:
        """
        Run BiCGStab iterations and update `out` in place.

        The method alternates between a BiCG-style search update and a local
        stabilization step based on the auxiliary vectors `s` and `t`.
        """
        ExecutorTolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        rho_previous = 1.0
        alpha = 1.0
        omega = 1.0

        for iteration in range(1, policy.max_iterations + 1):
            self._monitor_iteration_count = iteration
            rho = workspace.inner_product(self.shadow, self.residual)
            if abs(rho) <= policy.breakdown_tol:
                raise RuntimeError("BiCGStab broke down with a vanishing rho.")

            beta = (rho / rho_previous) * (alpha / omega)
            workspace.combine3_block(
                1.0,
                self.residual,
                beta,
                self.direction,
                -beta * omega,
                self.velocity,
                self.temporary,
            )
            workspace.copy_block(self.direction, self.temporary)

            self.preconditioner(self.direction, self.preconditioned_direction)
            operator(self.preconditioned_direction, self.velocity)
            denominator = workspace.inner_product(self.shadow, self.velocity)
            if abs(denominator) <= policy.breakdown_tol:
                raise RuntimeError("BiCGStab broke down with a singular shadow projection.")
            alpha = rho / denominator

            workspace.combine2_block(1.0, self.residual, -alpha, self.velocity, self.s_buffer)
            if ExecutorTolerance.accepts(workspace.norm(self.s_buffer), rhs_norm):
                workspace.combine2_block(1.0, out, alpha, self.preconditioned_direction, self.temporary)
                workspace.copy_block(out, self.temporary)
                residual_norm = workspace.norm(self.s_buffer)
                self._monitor_final_residual = residual_norm
                return iteration, residual_norm

            self.preconditioner(self.s_buffer, self.preconditioned_s)
            operator(self.preconditioned_s, self.t_buffer)
            tt = workspace.inner_product(self.t_buffer, self.t_buffer)
            if abs(tt) <= policy.breakdown_tol:
                raise RuntimeError("BiCGStab broke down with a vanishing t norm.")
            omega = workspace.inner_product(self.t_buffer, self.s_buffer) / tt
            if abs(omega) <= policy.breakdown_tol:
                raise RuntimeError("BiCGStab broke down with a vanishing omega.")

            workspace.combine3_block(
                1.0,
                out,
                alpha,
                self.preconditioned_direction,
                omega,
                self.preconditioned_s,
                self.temporary,
            )
            workspace.copy_block(out, self.temporary)
            workspace.combine2_block(1.0, self.s_buffer, -omega, self.t_buffer, self.residual)
            residual_norm = workspace.norm(self.residual)
            self._monitor_final_residual = residual_norm
            if ExecutorTolerance.accepts(residual_norm, rhs_norm):
                return iteration, residual_norm

            rho_previous = rho

        return policy.max_iterations, workspace.norm(self.residual)


__all__ = ["InverterBiCGStab"]










