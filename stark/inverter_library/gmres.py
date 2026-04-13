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
the Krylov support workers in `stark.inverter_support.krylov`.
"""

from stark.contracts import Block, InnerProduct, Workbench
from stark.safety import Safety
from stark.inverter_support.block_operator import BlockOperator
from stark.inverter_support.descriptor import InverterDescriptor
from stark.inverter_support.policy import InverterPolicy
from stark.inverter_support.krylov import Arnoldi, GivensRotations, HessenbergLeastSquares, NUMBA_AVAILABLE
from stark.inverter_support.tolerance import InverterTolerance
from stark.inverter_support.workspace import InverterWorkspace
from stark.tolerance import Tolerance


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
        self.arnoldi = Arnoldi(workspace, restart)
        self.rotations = GivensRotations(restart)
        self.least_squares = HessenbergLeastSquares(restart)

    def ensure_size(self, size: int) -> None:
        """Allocate or resize all cached blocks for a given block length."""
        if self.size == size:
            return
        workspace = self.workspace
        self.size = size
        self.applied = workspace.allocate_block(size)
        self.residual = workspace.allocate_block(size)
        self.correction = workspace.allocate_block(size)
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
        rhs_norm: float,
        tolerance: InverterTolerance,
        policy: InverterPolicy,
        remaining_iterations: int,
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
            self.arnoldi.build_column(
                column,
                operator,
                self.arnoldi.basis[column],
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
            workspace.combine2_block(temporary, 1.0, correction, coefficients[index], self.arnoldi.basis[index])
            workspace.copy_block(correction, temporary)

        workspace.combine2_block(temporary, 1.0, out, 1.0, correction)
        workspace.copy_block(out, temporary)


class InverterGMRES:
    """
    Restarted GMRES on STARK blocks using a user-supplied inner product.

    Conceptually this is the standard restarted GMRES method for solving

        A x = b

    with the following STARK substitutions:

    - `x` and `b` are `Block` objects rather than flat arrays
    - `A` is supplied as a `BlockOperator`
    - all vector-space mechanics are delegated to `InverterWorkspace`

    The inverter is configured once, bound to one operator, and then called
    repeatedly on preallocated blocks. This matches STARK's "configured worker"
    style and keeps the hot `__call__` path small.

    Reference:
        Saad and Schultz (1986), SIAM J. Sci. Stat. Comput. 7(3).
    """

    __slots__ = ("tolerance", "policy", "operator", "workspace", "cycle", "safety", "_call")

    descriptor = InverterDescriptor("GMRES", "Restarted GMRES")

    def __init__(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: InverterPolicy | None = None,
        safety: Safety | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else InverterTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else InverterPolicy()
        if self.policy.max_iterations < 1:
            raise ValueError("InverterPolicy.max_iterations must be at least 1.")
        if self.policy.restart < 1:
            raise ValueError("InverterPolicy.restart must be at least 1.")
        self.safety = safety if safety is not None else Safety()
        self.workspace = InverterWorkspace(workbench, translation_probe, inner_product, self.safety)
        self.operator: BlockOperator | None = None
        self.cycle = GMRESCycle(self.workspace, self.policy.restart)
        self._call = self._call_unbound

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return f"InverterGMRES(tolerance={self.tolerance!r}, policy={self.policy!r}, safety={self.safety!r})"

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
        raise RuntimeError("GMRES inverter must be bound to an operator before use.")

    def _call_safe(self, out: Block, rhs: Block) -> None:
        workspace = self.workspace
        workspace._check_size(out, rhs)
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
                rhs_norm,
                tolerance,
                policy,
                policy.max_iterations - iterations,
            )
            iterations += used_iterations
            if tolerance.accepts(beta, rhs_norm):
                return

        raise RuntimeError(
            f"{self.short_name} failed to converge within "
            f"{policy.max_iterations} iterations (residual={beta:g})."
        )


__all__ = ["InverterGMRES", "NUMBA_AVAILABLE"]
