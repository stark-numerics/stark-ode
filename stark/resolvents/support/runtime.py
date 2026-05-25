from __future__ import annotations

from stark.accelerators import AcceleratorAbsent
from stark.block import Block
from stark.contracts import AcceleratorLike, IntervalLike, State
from stark.execution.safety import Safety
from stark.resolvents.support.monitoring import MonitorResolventLike


def initialise_resolvent_runtime(
    resolvent,
    safety: Safety | None = None,
    accelerator: AcceleratorLike | None = None,
) -> None:
    resolvent.safety = safety if safety is not None else Safety()
    resolvent.accelerator = (
        accelerator if accelerator is not None else AcceleratorAbsent()
    )
    resolvent.alpha = 0.0
    resolvent._monitor = None

    if hasattr(resolvent, "call_inline"):
        resolvent.call_pure = resolvent.call_inline
        resolvent.redirect_call = resolvent.call_pure
    else:
        # Legacy fallback for resolvents not yet migrated.
        resolvent.interval = None
        resolvent.state = None
        resolvent.redirect_call = resolvent.call_unbound


def refresh_resolvent_call(resolvent) -> None:
    resolvent.redirect_call = resolvent.call_pure


def with_resolvent_display_methods(cls):
    """Install standard resolvent display and metadata methods."""

    @property
    def short_name(self) -> str:
        return self.descriptor.short_name

    def __repr__(self) -> str:
        extra_parts = []

        if hasattr(self, "depth"):
            extra_parts.append(f"depth={self.depth!r}")

        if hasattr(self, "inverter"):
            extra_parts.append(f"inverter={self.inverter!r}")

        extra = ", ".join(extra_parts)
        if extra:
            extra = f", {extra}"

        return (
            f"{type(self).__name__}("
            f"tolerance={self.tolerance!r}, "
            f"policy={self.policy!r}"
            f"{extra}, "
            f"accelerator={self.accelerator!r}, "
            f"tableau={self.tableau!r})"
        )

    def __str__(self) -> str:
        return type(self).__name__

    cls.short_name = short_name
    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls


def with_resolvent_call_methods(cls):
    """Install new-paradigm resolvent call routing."""

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator

    def __call__(self, problem, delta):
        return self.redirect_call(problem, delta)

    cls.bind_accelerator = bind_accelerator
    cls.__call__ = __call__
    return cls


def with_resolvent_binding_methods(cls):
    """Install legacy bound/unbound call routing.

    Kept only so old resolvents can coexist during staged migration.
    New resolvents should use ``with_resolvent_call_methods``.
    """

    def bind(self, interval: IntervalLike, state: State) -> None:
        self.interval = interval
        self.state = state
        self.redirect_call = (
            self.call_checked if self.safety.block_sizes else self.call_unchecked
        )

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.accelerator = accelerator

    def __call__(self, alpha: float, rhs: Block | None, out: Block) -> None:
        self.redirect_call(alpha, rhs, out)

    def call_unbound(self, alpha: float, rhs: Block | None, out: Block) -> None:
        del alpha, rhs, out
        raise RuntimeError(f"{type(self).__name__} must be bound before use.")

    cls.bind = bind
    cls.bind_accelerator = bind_accelerator
    cls.__call__ = __call__
    cls.call_unbound = call_unbound
    return cls


def with_resolvent_monitoring_methods(cls):
    """Install the standard resolvent monitor boundary."""

    def assign_monitor(self, monitor: MonitorResolventLike) -> None:
        self._monitor = monitor

    def unassign_monitor(self) -> None:
        self._monitor = None

    def record_solve(
        self,
        block_size: int,
        iteration_count: int,
        error: float,
        scale: float,
        converged: bool,
    ) -> None:
        monitor = self._monitor
        if monitor is None:
            return

        descriptor = getattr(type(self), "descriptor", None)
        resolvent_name = getattr(descriptor, "short_name", type(self).__name__)
        monitor.record_solve(
            resolvent_name,
            self.alpha,
            block_size,
            iteration_count,
            error,
            scale,
            converged,
        )

    cls.assign_monitor = assign_monitor
    cls.unassign_monitor = unassign_monitor
    cls.record_solve = record_solve
    return cls


def check_one_stage_block(name: str, block: Block) -> None:
    if len(block) != 1:
        raise ValueError(f"{name} must be a one-item block for this resolvent.")


__all__ = [
    "check_one_stage_block",
    "initialise_resolvent_runtime",
    "refresh_resolvent_call",
    "with_resolvent_binding_methods",
    "with_resolvent_call_methods",
    "with_resolvent_display_methods",
    "with_resolvent_monitoring_methods",
]
