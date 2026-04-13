from __future__ import annotations

"""
Flexible restarted GMRES for STARK block-valued linear systems.

FGMRES differs from ordinary GMRES by allowing the preconditioning operation to
change from one Krylov column to the next. This is the variant described by
Saad in:

    Y. Saad,
    "A Flexible Inner-Outer Preconditioned GMRES Algorithm",
    SIAM Journal on Scientific Computing 14(2), 1993.

In STARK terms, the "preconditioner" is just another inverter-like worker that
acts on `Block` objects. When no preconditioner is supplied, FGMRES reduces to
a slightly more general but somewhat more expensive GMRES-like scheme.
"""

from stark.contracts import Block, InverterLike, Workbench
from stark.contracts import InnerProduct
from stark.safety import Safety
from stark.inverter_support.block_operator import BlockOperator
from stark.inverter_support.descriptor import InverterDescriptor
from stark.inverter_support.policy import InverterPolicy
from stark.inverter_support.krylov import Arnoldi, GivensRotations, HessenbergLeastSquares, NUMBA_AVAILABLE
from stark.inverter_support.tolerance import InverterTolerance
from stark.inverter_support.workspace import InverterWorkspace
from stark.tolerance import Tolerance


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

    def __init__(self, workspace: InverterWorkspace, restart: int) -> None:
        self.workspace = workspace
        self.restart = restart
        self.size = -1
        self.applied = None
        self.residual = None
        self.correction = None
        self.search_basis = []
        self.arnoldi = Arnoldi(workspace, restart)
        self.rotations = GivensRotations(restart)
        self.least_squares = HessenbergLeastSquares(restart)

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


class InverterFGMRES:
    """
    Restarted flexible GMRES with an optional right preconditioner.

    This inverter solves `A x = b` over STARK blocks in the same general way as
    `InverterGMRES`, but it allows the preconditioning action to vary across the
    Krylov window. That makes it a good fit when the "preconditioner" is itself
    an iterative or stateful worker.

    In the current library the preconditioner slot accepts any `InverterLike`
    object. If no preconditioner is provided, FGMRES falls back to the identity
    action and behaves like an unpreconditioned flexible method.

    Reference:
        Saad (1993), SIAM J. Sci. Comput. 14(2).
    """

    __slots__ = (
        "tolerance",
        "policy",
        "operator",
        "workspace",
        "cycle",
        "safety",
        "preconditioner",
        "_call",
        "_apply_preconditioner",
    )

    descriptor = InverterDescriptor("FGMRES", "Flexible Restarted GMRES")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: InverterLike | None = None,
        safety: Safety | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else InverterTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else InverterPolicy()
        if self.policy.max_iterations < 1:
            raise ValueError("InverterPolicy.max_iterations must be at least 1.")
        if self.policy.restart < 1:
            raise ValueError("InverterPolicy.restart must be at least 1.")
        self.operator: BlockOperator | None = None
        self.safety = safety if safety is not None else Safety()
        self.workspace = InverterWorkspace(workbench, translation_probe, inner_product, self.safety)
        self.cycle = FGMRESCycle(self.workspace, self.policy.restart)
        self.preconditioner = preconditioner
        self._call = self._call_unbound
        self._apply_preconditioner = self._copy_preconditioner if preconditioner is None else preconditioner

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return (
            "InverterFGMRES("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}, "
            f"preconditioner={self.preconditioner!r}, "
            f"safety={self.safety!r})"
        )

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def bind(self, operator: BlockOperator) -> None:
        """Attach the operator and resolve the safe or fast call path."""
        self.operator = operator
        self._call = self._call_safe if self.safety.block_sizes else self._call_unsafe

    def prepare(self, size: int) -> None:
        """Allocate cached storage for blocks of a specific length."""
        self.cycle.ensure_size(size)

    def __call__(self, out: Block, rhs: Block) -> None:
        self._call(out, rhs)

    def _call_unbound(self, out: Block, rhs: Block) -> None:
        del out, rhs
        raise RuntimeError("FGMRES inverter must be bound to an operator before use.")

    def _call_safe(self, out: Block, rhs: Block) -> None:
        self.workspace._check_size(out, rhs)
        self.prepare(len(out))
        self._solve_prepared(out, rhs)

    def _call_unsafe(self, out: Block, rhs: Block) -> None:
        self.prepare(len(out))
        self._solve_prepared(out, rhs)

    def _solve_prepared(self, out: Block, rhs: Block) -> None:
        """Solve the prepared linear system in place on `out`."""
        operator = self.operator
        assert operator is not None
        tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        rhs_norm = workspace.norm(rhs)
        beta = self.cycle.initial_residual(out, rhs, operator)
        if tolerance.accepts(beta, rhs_norm):
            return

        iterations = 0
        while iterations < policy.max_iterations:
            used_iterations, beta = self.cycle.run(
                out,
                rhs,
                operator,
                tolerance,
                policy,
                rhs_norm,
                policy.max_iterations - iterations,
                self._apply_preconditioner,
            )
            iterations += used_iterations
            if tolerance.accepts(beta, rhs_norm):
                return

        raise RuntimeError(
            f"{self.short_name} failed to converge within "
            f"{policy.max_iterations} iterations (residual={beta:g})."
        )

    def _copy_preconditioner(self, out: Block, rhs: Block) -> None:
        """Identity preconditioner used when no external worker is supplied."""
        self.workspace.copy_block(out, rhs)


__all__ = ["InverterFGMRES", "NUMBA_AVAILABLE"]
