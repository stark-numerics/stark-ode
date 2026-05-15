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

from stark.contracts import AcceleratorLike, Block, InnerProduct, PreconditionerLike, Workbench
from stark.block.operator import BlockOperator
from stark.execution.safety import Safety
from stark.inverters.base import InverterBase
from stark.inverters.descriptor import InverterDescriptor
from stark.inverters.policy import InverterPolicy
from stark.inverters.tolerance import InverterTolerance
from stark.execution.tolerance import Tolerance


class BiCGStabCycle:
    """
    Reusable BiCGStab scratch storage for one operator binding.

    The cycle owns the extra residual-like blocks needed by the short-recurrence
    update:

    - `residual`: the current residual
    - `shadow`: the fixed BiCG shadow residual
    - `direction`: the current search direction
    - `velocity`: `A direction`
    - `s_buffer`, `t_buffer`: the stabilization buffers
    """

    __slots__ = (
        "workspace",
        "size",
        "applied",
        "residual",
        "shadow",
        "direction",
        "velocity",
        "preconditioned_direction",
        "preconditioned_s",
        "s_buffer",
        "t_buffer",
        "temporary",
    )

    def __init__(self, workspace: InverterWorkspace) -> None:
        self.workspace = workspace
        self.size = -1
        self.applied = None
        self.residual = None
        self.shadow = None
        self.direction = None
        self.velocity = None
        self.preconditioned_direction = None
        self.preconditioned_s = None
        self.s_buffer = None
        self.t_buffer = None
        self.temporary = None

    def ensure_size(self, size: int) -> None:
        """Allocate or resize all cached blocks for a given block length."""
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

    def initial_residual(self, rhs: Block, out: Block, operator: BlockOperator) -> float:
        """
        Compute `r = rhs - A out`, initialize the BiCG shadow state, and return
        the residual norm.
        """
        workspace = self.workspace
        applied = self.applied
        residual = self.residual
        shadow = self.shadow
        direction = self.direction
        velocity = self.velocity
        assert applied is not None
        assert residual is not None
        assert shadow is not None
        assert direction is not None
        assert velocity is not None
        operator(out, applied)
        workspace.combine2_block(residual, 1.0, rhs, -1.0, applied)
        workspace.copy_block(shadow, residual)
        workspace.zero_block(direction)
        workspace.zero_block(velocity)
        return workspace.norm(residual)

    def iterate(
        self,
        rhs: Block,
        out: Block,
        operator: BlockOperator,
        tolerance: InverterTolerance,
        policy: InverterPolicy,
        rhs_norm: float,
        apply_preconditioner,
    ) -> float:
        """
        Run BiCGStab iterations and update `out` in place.

        The method alternates between a BiCG-style search update and a local
        stabilization step based on the auxiliary vectors `s` and `t`.
        """
        workspace = self.workspace
        residual = self.residual
        shadow = self.shadow
        direction = self.direction
        velocity = self.velocity
        preconditioned_direction = self.preconditioned_direction
        preconditioned_s = self.preconditioned_s
        s_buffer = self.s_buffer
        t_buffer = self.t_buffer
        temporary = self.temporary
        assert residual is not None
        assert shadow is not None
        assert direction is not None
        assert velocity is not None
        assert preconditioned_direction is not None
        assert preconditioned_s is not None
        assert s_buffer is not None
        assert t_buffer is not None
        assert temporary is not None

        rho_previous = 1.0
        alpha = 1.0
        omega = 1.0

        for _ in range(policy.max_iterations):
            rho = workspace.inner_product(shadow, residual)
            if abs(rho) <= policy.breakdown_tol:
                raise RuntimeError(f"{InverterBiCGStab.descriptor.short_name} broke down with a vanishing rho.")

            beta = (rho / rho_previous) * (alpha / omega)
            workspace.combine3_block(temporary, 1.0, residual, beta, direction, -beta * omega, velocity)
            workspace.copy_block(direction, temporary)

            apply_preconditioner(direction, preconditioned_direction)
            operator(preconditioned_direction, velocity)
            denominator = workspace.inner_product(shadow, velocity)
            if abs(denominator) <= policy.breakdown_tol:
                raise RuntimeError(
                    f"{InverterBiCGStab.descriptor.short_name} broke down with a singular shadow projection."
                )
            alpha = rho / denominator

            workspace.combine2_block(s_buffer, 1.0, residual, -alpha, velocity)
            if tolerance.accepts(workspace.norm(s_buffer), rhs_norm):
                workspace.combine2_block(temporary, 1.0, out, alpha, preconditioned_direction)
                workspace.copy_block(out, temporary)
                return workspace.norm(s_buffer)

            apply_preconditioner(s_buffer, preconditioned_s)
            operator(preconditioned_s, t_buffer)
            tt = workspace.inner_product(t_buffer, t_buffer)
            if abs(tt) <= policy.breakdown_tol:
                raise RuntimeError(f"{InverterBiCGStab.descriptor.short_name} broke down with a vanishing t norm.")
            omega = workspace.inner_product(t_buffer, s_buffer) / tt
            if abs(omega) <= policy.breakdown_tol:
                raise RuntimeError(f"{InverterBiCGStab.descriptor.short_name} broke down with a vanishing omega.")

            workspace.combine3_block(temporary, 1.0, out, alpha, preconditioned_direction, omega, preconditioned_s)
            workspace.copy_block(out, temporary)
            workspace.combine2_block(residual, 1.0, s_buffer, -omega, t_buffer)
            residual_norm = workspace.norm(residual)
            if tolerance.accepts(residual_norm, rhs_norm):
                return residual_norm

            rho_previous = rho

        return workspace.norm(residual)


class InverterBiCGStab(InverterBase):
    """
    Operator-based BiCGStab on STARK blocks.

    This inverter solves `A x = b` using a short-recurrence Krylov method that
    only needs a fixed amount of block storage, independent of the iteration
    count. That makes it cheaper in memory than GMRES-family methods, at the
    cost of somewhat rougher convergence behaviour.

    Reference:
        van der Vorst (1992), SIAM J. Sci. Stat. Comput. 13(2).
    """

    __slots__ = ()

    descriptor = InverterDescriptor("BiCGStab", "BiConjugate Gradient Stabilized")

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

    def make_cycle(self) -> BiCGStabCycle:
        return BiCGStabCycle(self.workspace)

    def solve_prepared(self, rhs: Block, out: Block) -> None:
        operator = self.operator
        assert operator is not None
        tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        rhs_norm = workspace.norm(rhs)
        residual_norm = self.cycle.initial_residual(rhs, out, operator)
        if tolerance.accepts(residual_norm, rhs_norm):
            return

        residual_norm = self.cycle.iterate(rhs, out, operator, tolerance, policy, rhs_norm, self.preconditioner)
        if tolerance.accepts(residual_norm, rhs_norm):
            return

        raise RuntimeError(
            f"{self.short_name} failed to converge within "
            f"{policy.max_iterations} iterations (residual={residual_norm:g})."
        )


__all__ = ["InverterBiCGStab"]










