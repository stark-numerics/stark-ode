from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from stark.accelerators import AcceleratorAbsent
from stark.block.operator import BlockOperator
from stark.contracts import AcceleratorLike, Block, InnerProduct, PreconditionerLike, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.inverters.policy import InverterPolicy
from stark.inverters.support.preconditioner import Preconditioner
from stark.inverters.support.workspace import InverterWorkspace
from stark.inverters.tolerance import InverterTolerance


class InverterBase(ABC):
    __slots__ = ("tolerance", "policy", "operator", "workspace", "cycle", "safety", "accelerator", "preconditioner", "redirect_call")

    descriptor: object

    def initialise_inverter(
        self,
        workbench: Workbench,
        inner_product: InnerProduct,
        tolerance: Tolerance | None = None,
        policy: InverterPolicy | None = None,
        preconditioner: PreconditionerLike | None = None,
        safety: Safety | None = None,
        accelerator: AcceleratorLike | None = None,
    ) -> None:
        translation_probe = workbench.allocate_translation()
        self.tolerance = tolerance if tolerance is not None else InverterTolerance(atol=1.0e-9, rtol=1.0e-9)
        self.policy = policy if policy is not None else InverterPolicy()
        self.validate_policy()
        self.operator = None
        self.safety = safety if safety is not None else Safety()
        self.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
        self.workspace = InverterWorkspace(
            workbench,
            translation_probe,
            inner_product,
            self.safety,
            accelerator=self.accelerator,
        )
        self.preconditioner = Preconditioner(self.workspace, preconditioner)
        self.cycle = self.make_cycle()
        self.redirect_call = self.call_unbound

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    @property
    def full_name(self) -> str:
        return self.descriptor.full_name

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}, "
            f"preconditioner={self.preconditioner!r}, "
            f"safety={self.safety!r}, "
            f"accelerator={self.accelerator!r})"
        )

    def __str__(self) -> str:
        return f"{self.short_name} with {self.tolerance}, {self.policy}"

    def bind(self, operator: BlockOperator) -> None:
        self.operator = operator
        self.preconditioner.bind(operator)
        self.redirect_call = self.call_checked if self.safety.block_sizes else self.call_unchecked

    def prepare(self, size: int) -> None:
        self.cycle.ensure_size(size)
        self.preconditioner.prepare(size)

    def __call__(self, out: Block, rhs: Block) -> None:
        self.redirect_call(out, rhs)

    def call_unbound(self, out: Block, rhs: Block) -> None:
        del out, rhs
        raise RuntimeError(f"{self.short_name} inverter must be bound to an operator before use.")

    def call_checked(self, out: Block, rhs: Block) -> None:
        self.workspace._check_size(out, rhs)
        self.prepare(len(out))
        self.solve_prepared(out, rhs)

    def call_unchecked(self, out: Block, rhs: Block) -> None:
        self.prepare(len(out))
        self.solve_prepared(out, rhs)

    def validate_policy(self) -> None:
        if self.policy.max_iterations < 1:
            raise ValueError("InverterPolicy.max_iterations must be at least 1.")

    @abstractmethod
    def make_cycle(self) -> Any:
        """Construct the concrete iteration worker."""

    @abstractmethod
    def solve_prepared(self, out: Block, rhs: Block) -> None:
        """Solve the prepared linear system in place on `out`."""


class InverterBaseRestartedKrylov(InverterBase):
    def validate_policy(self) -> None:
        super().validate_policy()
        if self.policy.restart < 1:
            raise ValueError("InverterPolicy.restart must be at least 1.")

    def solve_prepared(self, out: Block, rhs: Block) -> None:
        operator = self.operator
        assert operator is not None
        tolerance = self.tolerance
        policy = self.policy
        workspace = self.workspace
        cycle = self.cycle
        rhs_norm = workspace.norm(rhs)
        residual_norm = cycle.initial_residual(out, rhs, operator)
        if tolerance.accepts(residual_norm, rhs_norm):
            return

        iterations = 0
        while iterations < policy.max_iterations:
            used_iterations, residual_norm = cycle.run(
                out,
                rhs,
                operator,
                tolerance,
                policy,
                rhs_norm,
                policy.max_iterations - iterations,
                self.preconditioner,
            )
            iterations += used_iterations
            if tolerance.accepts(residual_norm, rhs_norm):
                return

        raise RuntimeError(
            f"{self.short_name} failed to converge within "
            f"{policy.max_iterations} iterations (residual={residual_norm:g})."
        )


__all__ = ["InverterBase", "InverterBaseRestartedKrylov"]
