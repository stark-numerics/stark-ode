from __future__ import annotations

from stark.accelerators import AcceleratorAbsent
from stark.block.operator import BlockOperator
from stark.contracts import AcceleratorLike, Block, InnerProduct, PreconditionerLike, Workbench
from stark.execution.safety import Safety
from stark.execution.tolerance import Tolerance
from stark.inverters.support.policy import InverterPolicy
from stark.inverters.support.preconditioner import Preconditioner
from stark.inverters.support.tolerance import InverterTolerance
from stark.inverters.support.workspace import InverterWorkspace


def initialise_inverter_runtime(
    inverter,
    workbench: Workbench,
    inner_product: InnerProduct,
    tolerance: Tolerance | None = None,
    policy: InverterPolicy | None = None,
    preconditioner: PreconditionerLike | None = None,
    safety: Safety | None = None,
    accelerator: AcceleratorLike | None = None,
) -> None:
    translation_probe = workbench.allocate_translation()
    inverter.tolerance = tolerance if tolerance is not None else InverterTolerance(atol=1.0e-9, rtol=1.0e-9)
    inverter.policy = policy if policy is not None else InverterPolicy()
    inverter.operator = None
    inverter.safety = safety if safety is not None else Safety()
    inverter.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
    inverter.workspace = InverterWorkspace(
        workbench,
        translation_probe,
        inner_product,
        inverter.safety,
        accelerator=inverter.accelerator,
    )
    inverter.preconditioner = Preconditioner(inverter.workspace, preconditioner)
    inverter.redirect_call = inverter.call_unbound


def validate_inverter_policy(policy: InverterPolicy) -> None:
    if policy.max_iterations < 1:
        raise ValueError("InverterPolicy.max_iterations must be at least 1.")


def validate_restarted_inverter_policy(policy: InverterPolicy) -> None:
    validate_inverter_policy(policy)
    if policy.restart < 1:
        raise ValueError("InverterPolicy.restart must be at least 1.")


def with_inverter_display_methods(cls):
    """Install standard inverter display and metadata methods."""

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

    cls.short_name = short_name
    cls.full_name = full_name
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls


def with_inverter_binding_methods(cls):
    """Install standard bound/unbound inverter call routing methods."""

    def bind(self, operator: BlockOperator) -> None:
        self.operator = operator
        self.preconditioner.bind(operator)
        self.redirect_call = self.call_checked if self.safety.block_sizes else self.call_unchecked

    def prepare(self, size: int) -> None:
        self.ensure_size(size)
        self.preconditioner.prepare(size)

    def __call__(self, rhs: Block, out: Block) -> None:
        self.redirect_call(rhs, out)

    def call_unbound(self, rhs: Block, out: Block) -> None:
        del rhs, out
        raise RuntimeError(f"{self.short_name} inverter must be bound to an operator before use.")

    def call_checked(self, rhs: Block, out: Block) -> None:
        self.workspace._check_size(out, rhs)
        self.prepare(len(out))
        self.solve_prepared(rhs, out)

    def call_unchecked(self, rhs: Block, out: Block) -> None:
        self.prepare(len(out))
        self.solve_prepared(rhs, out)

    cls.bind = bind
    cls.prepare = prepare
    cls.__call__ = __call__
    cls.call_unbound = call_unbound
    cls.call_checked = call_checked
    cls.call_unchecked = call_unchecked
    return cls


__all__ = [
    "initialise_inverter_runtime",
    "validate_inverter_policy",
    "validate_restarted_inverter_policy",
    "with_inverter_binding_methods",
    "with_inverter_display_methods",
]
