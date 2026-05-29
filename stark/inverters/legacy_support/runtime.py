from __future__ import annotations

from stark.accelerators import AcceleratorAbsent
from stark.block import Block
from stark.block.operator import BlockOperator
from stark.contracts import AcceleratorLike, InnerProduct, InverterPreconditionerLike, Allocator
from stark.executor.tolerance import ExecutorTolerance
from stark.inverters.legacy_support.monitoring import MonitorInverterLike
from stark.inverters.legacy_support.policy import InverterPolicy
from stark.inverters.legacy_support.preconditioner import InverterPreconditioner
from stark.inverters.legacy_support.safety import InverterSafety, InverterSafetyDefault
from stark.inverters.legacy_support.tolerance import InverterTolerance
from stark.inverters.legacy_support.workspace import InverterWorkspace


def initialise_inverter_runtime(
    inverter,
    allocator: Allocator,
    inner_product: InnerProduct,
    tolerance: ExecutorTolerance | None = None,
    policy: InverterPolicy | None = None,
    preconditioner: InverterPreconditionerLike | None = None,
    safety: InverterSafety | None = None,
    accelerator: AcceleratorLike | None = None,
) -> None:
    translation_probe = allocator.allocate_translation()
    inverter.tolerance = tolerance if tolerance is not None else InverterTolerance(atol=1.0e-9, rtol=1.0e-9)
    inverter.policy = policy if policy is not None else InverterPolicy()
    inverter.operator = None
    inverter.safety = safety if safety is not None else InverterSafetyDefault()
    inverter.accelerator = accelerator if accelerator is not None else AcceleratorAbsent()
    inverter.workspace = InverterWorkspace(
        allocator,
        translation_probe,
        inner_product,
        inverter.safety,
        accelerator=inverter.accelerator,
    )
    inverter.preconditioner = InverterPreconditioner(inverter.workspace, preconditioner)
    inverter._monitor = None
    inverter._bound_call = inverter.call_unbound
    inverter._monitor_iteration_count = None
    inverter._monitor_initial_residual = None
    inverter._monitor_final_residual = None
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

    def refresh_call(self) -> None:
        if self.operator is None:
            self.redirect_call = self.call_unbound
            return
        self.redirect_call = self.call_monitored if self._monitor is not None else self._bound_call

    def bind(self, operator: BlockOperator) -> None:
        self.operator = operator
        self.preconditioner.bind(operator)
        self._bound_call = self.call_checked if self.safety.block_sizes else self.call_unchecked
        self.refresh_call()

    def assign_monitor(self, monitor: MonitorInverterLike) -> None:
        self._monitor = monitor
        self.refresh_call()

    def unassign_monitor(self) -> None:
        self._monitor = None
        self.refresh_call()

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

    def call_monitored(self, rhs: Block, out: Block) -> None:
        monitor = self._monitor
        try:
            self._bound_call(rhs, out)
        except Exception as error:
            if monitor is not None:
                failure_reason = str(error) or type(error).__name__
                monitor.record_solve(
                    self.short_name,
                    False,
                    self._monitor_iteration_count,
                    self._monitor_initial_residual,
                    self._monitor_final_residual,
                    failure_reason,
                )
            raise

        if monitor is not None:
            monitor.record_solve(
                self.short_name,
                True,
                self._monitor_iteration_count,
                self._monitor_initial_residual,
                self._monitor_final_residual,
                None,
            )

    cls.refresh_call = refresh_call
    cls.bind = bind
    cls.assign_monitor = assign_monitor
    cls.unassign_monitor = unassign_monitor
    cls.prepare = prepare
    cls.__call__ = __call__
    cls.call_unbound = call_unbound
    cls.call_checked = call_checked
    cls.call_unchecked = call_unchecked
    cls.call_monitored = call_monitored
    return cls


__all__ = [
    "initialise_inverter_runtime",
    "validate_inverter_policy",
    "validate_restarted_inverter_policy",
    "with_inverter_binding_methods",
    "with_inverter_display_methods",
]
