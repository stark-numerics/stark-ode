"""Single-step orchestration for schemes and executors."""

from __future__ import annotations

from stark.core.auditor import Auditor
from stark.executor.executor import Executor
from stark.contracts import IntervalLike, SchemeLike, State


class Marcher:
    __slots__ = ("scheme", "executor")

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








