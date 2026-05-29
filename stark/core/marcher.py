"""Single-step orchestration for schemes, executors, and monitors."""

from __future__ import annotations

from stark.core.auditor import Auditor
from stark.executor.executor import Executor
from stark.contracts import IntervalLike, SchemeLike, State
from stark.monitor import Monitor


_MONITOR_DESCENDANT_ATTRIBUTES = ("stepper", "stage_solver", "resolvent", "inverter")


def _assign_descendant_monitors(worker: object, monitor: Monitor, visited: set[int]) -> None:
    worker_id = id(worker)
    if worker_id in visited:
        return
    visited.add(worker_id)

    for name in _MONITOR_DESCENDANT_ATTRIBUTES:
        child = getattr(worker, name, None)
        if child is None:
            continue
        if name == "resolvent":
            assign_monitor = getattr(child, "assign_monitor", None)
            if callable(assign_monitor):
                assign_monitor(monitor.resolvent)
        elif name == "inverter":
            assign_monitor = getattr(child, "assign_monitor", None)
            if callable(assign_monitor):
                assign_monitor(monitor.inverter)
        _assign_descendant_monitors(child, monitor, visited)


def _unassign_descendant_monitors(worker: object, visited: set[int]) -> None:
    worker_id = id(worker)
    if worker_id in visited:
        return
    visited.add(worker_id)

    for name in _MONITOR_DESCENDANT_ATTRIBUTES:
        child = getattr(worker, name, None)
        if child is None:
            continue
        unassign_monitor = getattr(child, "unassign_monitor", None)
        if callable(unassign_monitor):
            unassign_monitor()
        _unassign_descendant_monitors(child, visited)


class Marcher:
    __slots__ = ("scheme", "executor", "monitor")

    def __init__(
        self,
        scheme: SchemeLike,
        executor: Executor,
    ) -> None:
        if not isinstance(executor, Executor):
            raise TypeError("Marcher requires an Executor.")
        Auditor.require_marcher_inputs(scheme, executor.tolerance, executor.safety)
        self.scheme = scheme
        self.executor = executor
        self.monitor: Monitor | None = None
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

        accepted_dt = self.scheme(interval, state, self.executor)
        interval.increment(accepted_dt)

    def snapshot_state(self, state: State) -> State:
        return self.scheme.snapshot_state(state)

    def set_executor(self, executor: Executor) -> None:
        if not isinstance(executor, Executor):
            raise TypeError("Marcher.set_executor(...) requires an Executor.")
        self.executor = executor
        self.scheme.set_apply_delta_safety(self.executor.safety.apply_delta)

    def assign_monitor(self, monitor: Monitor) -> None:
        if not hasattr(self.scheme, "monitor") or not hasattr(self.scheme, "call_monitored"):
            raise TypeError("Marcher scheme does not support monitoring.")

        self.monitor = monitor
        self.scheme.monitor = monitor.scheme
        self.scheme.call_step = self.scheme.call_monitored
        self.scheme.redirect_call = self.scheme.call_step
        _assign_descendant_monitors(self.scheme, monitor, set())

    def unassign_monitor(self) -> None:
        if hasattr(self.scheme, "monitor"):
            self.scheme.monitor = None
        if hasattr(self.scheme, "call_body"):
            self.scheme.call_step = self.scheme.call_body
            self.scheme.redirect_call = self.scheme.call_step
        _unassign_descendant_monitors(self.scheme, set())
        self.monitor = None











