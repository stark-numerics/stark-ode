from __future__ import annotations

from stark.accelerators.binding import bind_worker_tree
from stark.auditor import Auditor
from stark.execution.executor import Executor
from stark.contracts import IntervalLike, SchemeLike, State
from stark.execution.safety import Safety
from stark.monitor import Monitor


class Marcher:
    __slots__ = ("scheme", "executor", "monitor")

    def __init__(
        self,
        scheme: SchemeLike,
        executor: Executor,
    ) -> None:
        if not isinstance(executor, Executor):
            raise TypeError("Marcher requires an Executor.")
        Auditor.require_marcher_inputs(scheme, executor.tolerance, executor.safety, executor.accelerator)
        self.scheme = scheme
        self.executor = executor
        self.monitor: Monitor | None = None
        bind_worker_tree(self.scheme, self.executor.accelerator)
        assign_executor = getattr(self.scheme, "assign_executor", None)
        if callable(assign_executor):
            assign_executor(self.executor)
        self.scheme.set_apply_delta_safety(executor.safety.apply_delta)

    def __repr__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return (
            "Marcher("
            f"scheme={scheme_name!r}, "
            f"executor={self.executor!r})"
        )

    def __str__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return f"Marcher {scheme_name} with {self.executor.tolerance}"

    def __call__(self, interval: IntervalLike, state: State) -> None:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return

        self.scheme.set_apply_delta_safety(self.executor.safety.apply_delta)
        accepted_dt = self.scheme(interval, state, self.executor)
        interval.increment(accepted_dt)

    def snapshot_state(self, state: State) -> State:
        return self.scheme.snapshot_state(state)

    def set_executor(self, executor: Executor) -> None:
        if not isinstance(executor, Executor):
            raise TypeError("Marcher.set_executor(...) requires an Executor.")
        self.executor = executor
        bind_worker_tree(self.scheme, self.executor.accelerator)
        assign_executor = getattr(self.scheme, "assign_executor", None)
        if callable(assign_executor):
            assign_executor(self.executor)
        self.scheme.set_apply_delta_safety(self.executor.safety.apply_delta)

    def set_safety(self, safety: Safety) -> None:
        self.executor = self.executor.with_updates(safety=safety)
        self.scheme.set_apply_delta_safety(safety.apply_delta)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        safety = self.executor.safety
        self.set_safety(Safety(progress=safety.progress, block_sizes=safety.block_sizes, apply_delta=enabled))

    def assign_monitor(self, monitor: Monitor) -> None:
        assign_monitor = getattr(self.scheme, "assign_monitor", None)
        if not callable(assign_monitor):
            raise TypeError("Marcher scheme does not support monitoring.")
        self.monitor = monitor
        assign_monitor(monitor)

    def unassign_monitor(self) -> None:
        unassign_monitor = getattr(self.scheme, "unassign_monitor", None)
        if callable(unassign_monitor):
            unassign_monitor()
        self.monitor = None











