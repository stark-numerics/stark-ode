from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from stark.auditor import Auditor
from stark.execution.executor import Executor
from stark.marcher import Marcher
from stark.contracts import IntervalLike, State
from stark.execution.safety import Safety
from stark.monitor import Monitor


Checkpoints = int | Iterable[float]


class Integrator:
    """
    Chain `Marcher` calls until the interval reaches its stop time.

    Calling the instance itself yields snapshots: each `(interval, state)` pair
    is copied before it is yielded, so collecting the iterator gives a real
    trajectory rather than repeated references to the same mutable objects.

    Call `live(...)` for the zero-copy path. That yields the original mutable
    interval and state objects after each step, which is better for benchmarks
    and tight loops but means every yielded pair refers to the same evolving
    objects.

    Both modes accept optional checkpoints. `checkpoints=4` over `[0, 1]`
    yields only at `0.25`, `0.5`, `0.75`, and `1.0`; an explicit iterable gives
    absolute checkpoint times, and the final interval stop is appended if it is
    not already present. Internally this runs the existing stepping loop over
    consecutive subintervals, so adaptive schemes land exactly on each
    checkpoint.

    Snapshot mode relies on explicit copy support: intervals must implement
    `copy()`, and `Marcher` provides state snapshots through its scheme
    workbench rather than falling back to `deepcopy`.

    When progress safety is enabled, both modes check that time advances after
    every accepted step and raise a helpful error if a scheme stalls.
    """

    __slots__ = ("executor", "redirect_call_snapshot", "redirect_call_live")

    def __init__(
        self,
        executor: Executor | None = None,
        *,
        safety: Safety | None = None,
    ) -> None:
        self.executor = Executor.resolve(executor, safety=safety)
        self.redirect_call_snapshot = self.call_snapshot_safe if self.executor.safety.progress else self.call_snapshot_fast
        self.redirect_call_live = self.call_live_safe if self.executor.safety.progress else self.call_live_fast

    def __repr__(self) -> str:
        return f"Integrator(executor={self.executor!r})"

    def __str__(self) -> str:
        mode = "safe" if self.executor.safety.progress else "fast"
        return f"STARK integrator ({mode} mode)"

    def __call__(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield snapshot copies after each accepted step, or only at checkpoints.
        """
        Auditor.require_integration_inputs(marcher, interval, state, snapshots=True)
        if checkpoints is not None:
            return self.call_snapshot_checkpoints(marcher, interval, state, checkpoints)
        return self.redirect_call_snapshot(marcher, interval, state)

    def live(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield live mutable objects after each accepted step, or only at checkpoints.
        """
        Auditor.require_integration_inputs(marcher, interval, state, snapshots=False)
        if checkpoints is not None:
            return self.call_live_checkpoints(marcher, interval, state, checkpoints)
        return self.redirect_call_live(marcher, interval, state)

    def monitored(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        monitor: Monitor,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        with self.monitoring(marcher, monitor):
            yield from self(marcher, interval, state, checkpoints=checkpoints)

    def live_monitored(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        monitor: Monitor,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        with self.monitoring(marcher, monitor):
            yield from self.live(marcher, interval, state, checkpoints=checkpoints)

    def snapshot(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> tuple[IntervalLike, State]:
        return interval.copy(), marcher.snapshot_state(state)

    def call_snapshot_checkpoints(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        copy_interval = interval.copy
        snapshot_state = marcher.snapshot_state
        for checkpoint in self.call_live_checkpoints(marcher, interval, state, checkpoints):
            del checkpoint
            yield copy_interval(), snapshot_state(state)

    def call_live_checkpoints(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        targets = self.checkpoint_targets(interval, checkpoints)
        live_impl = self.redirect_call_live
        same_time = self.same_time
        original_stop = interval.stop
        last_positive_step = interval.step if self.advances_time(interval.present, interval.step) else None
        try:
            for index, target in enumerate(targets):
                interval.stop = target
                remaining = target - interval.present
                if interval.step <= 0.0 or not self.advances_time(interval.present, min(interval.step, remaining)):
                    interval.step = min(last_positive_step, remaining) if last_positive_step is not None else remaining
                for checkpoint in live_impl(marcher, interval, state):
                    del checkpoint
                    pass
                if not same_time(interval.present, target):
                    raise RuntimeError(
                        "Integration did not land on the requested checkpoint. "
                        "Use Marcher(...) or a checkpoint-aware marcher that clamps to interval.stop."
                    )
                if self.advances_time(interval.present, interval.step):
                    last_positive_step = interval.step
                yield interval, state
        finally:
            interval.stop = original_stop

    @classmethod
    def checkpoint_targets(cls, interval: IntervalLike, checkpoints: Checkpoints) -> tuple[float, ...]:
        start = float(interval.present)
        stop = float(interval.stop)
        if stop < start:
            raise ValueError("Interval stop must be greater than or equal to present.")
        if cls.same_time(start, stop):
            return ()

        if isinstance(checkpoints, int):
            if checkpoints < 1:
                raise ValueError("Integer checkpoints must be at least 1.")
            width = (stop - start) / checkpoints
            targets = [start + width * index for index in range(1, checkpoints + 1)]
            targets[-1] = stop
            return tuple(targets)

        targets = [float(checkpoint) for checkpoint in checkpoints]
        previous = start
        for target in targets:
            if cls.same_time(target, previous) or target < previous:
                raise ValueError("Checkpoints must be strictly increasing and after interval.present.")
            if target > stop and not cls.same_time(target, stop):
                raise ValueError("Checkpoints must not exceed interval.stop.")
            previous = target

        if not targets or not cls.same_time(targets[-1], stop):
            targets.append(stop)
        else:
            targets[-1] = stop
        return tuple(targets)

    @staticmethod
    def same_time(left: float, right: float) -> bool:
        return abs(left - right) <= 1.0e-12 * max(1.0, abs(left), abs(right))

    @staticmethod
    def advances_time(present: float, step: float) -> bool:
        return step > 0.0 and present + step > present

    @staticmethod
    @contextmanager
    def monitoring(marcher: Marcher, monitor: Monitor) -> Iterator[None]:
        marcher.assign_monitor(monitor)
        try:
            yield
        finally:
            marcher.unassign_monitor()

    def call_live_fast(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        march = marcher
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            march(interval, state)
            yield interval, state

    def call_live_safe(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        march = marcher
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            previous_present = interval.present
            march(interval, state)
            if interval.present <= previous_present:
                if same_time(interval.present, interval.stop) or same_time(previous_present, interval.stop):
                    interval.present = interval.stop
                    break
                raise RuntimeError(
                    "Integration made no progress. "
                    "The scheme may have returned a non-positive step size. "
                    "Disable progress safety only if you intentionally want unchecked behavior."
                )
            yield interval, state

    def call_snapshot_fast(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        march = marcher
        copy_interval = interval.copy
        snapshot_state = marcher.snapshot_state
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            march(interval, state)
            yield copy_interval(), snapshot_state(state)

    def call_snapshot_safe(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        march = marcher
        copy_interval = interval.copy
        snapshot_state = marcher.snapshot_state
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            previous_present = interval.present
            march(interval, state)
            if interval.present <= previous_present:
                if same_time(interval.present, interval.stop) or same_time(previous_present, interval.stop):
                    interval.present = interval.stop
                    break
                raise RuntimeError(
                    "Integration made no progress. "
                    "The scheme may have returned a non-positive step size. "
                    "Disable progress safety only if you intentionally want unchecked behavior."
                )
            yield copy_interval(), snapshot_state(state)

__all__ = ["Checkpoints", "Integrator"]










