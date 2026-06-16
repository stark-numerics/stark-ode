"""Repeated integration loops over a configured stepper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from stark.core.integrator.configuration import IntegratorConfiguration, IntegratorConfigurationDefault
from stark.core.auditor import Auditor
from stark.core.integrator.stepper import IntegratorStepper
from stark.core.contracts import IntervalLike, State


Checkpoints = int | Iterable[float]


class Integrator:
    """
    Chain `IntegratorStepper` calls until the interval reaches its stop time.

    Calling the instance itself yields snapshots: each `(interval, state)` pair
    is copied before it is yielded, so collecting the iterator gives a real
    trajectory rather than repeated references to the same mutable objects.

    Call `mutating_trajectory(...)` for the zero-copy path. That yields the
    original mutable interval and state objects after each step, which is better
    for benchmarks and tight loops but means every yielded pair refers to the
    same evolving objects.

    Both modes accept optional checkpoints. `checkpoints=4` over `[0, 1]`
    yields only at `0.25`, `0.5`, `0.75`, and `1.0`; an explicit iterable gives
    absolute checkpoint times, and the final interval stop is appended if it is
    not already present. Internally this runs the existing stepping loop over
    consecutive subintervals, so adaptive schemes land exactly on each
    checkpoint.

    Snapshot mode relies on explicit copy support: intervals must implement
    `copy()`, and `IntegratorStepper` provides state snapshots through its scheme
    allocator rather than falling back to `deepcopy`.

    When progress safety is enabled, both modes check that time advances after
    every accepted step and raise a helpful error if a scheme stalls.
    """

    __slots__ = ("configuration", "redirect_stable_trajectory", "redirect_mutating_trajectory")

    def __init__(
        self,
        configuration: IntegratorConfiguration | None = None,
    ) -> None:
        self.configuration = configuration if configuration is not None else IntegratorConfigurationDefault()
        self.redirect_stable_trajectory = (
            self.stable_trajectory_safe
            if self.configuration.check_progress
            else self.stable_trajectory_fast
        )
        self.redirect_mutating_trajectory = (
            self.mutating_trajectory_safe
            if self.configuration.check_progress
            else self.mutating_trajectory_fast
        )

    def __repr__(self) -> str:
        return f"Integrator(configuration={self.configuration!r})"

    def __str__(self) -> str:
        mode = "safe" if self.configuration.check_progress else "fast"
        return f"STARK integrator ({mode} mode)"

    def __call__(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        return self.stable_trajectory(stepper, interval, state, checkpoints)

    def stable_trajectory(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield snapshot copies after each accepted step, or only at checkpoints.
        """
        Auditor.require_integration_inputs(stepper, interval, state, snapshots=True)
        if checkpoints is not None:
            return self.stable_trajectory_checkpoints(stepper, interval, state, checkpoints)
        return self.redirect_stable_trajectory(stepper, interval, state)

    def mutating_trajectory(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints | None = None,
    ) -> Iterator[tuple[IntervalLike, State]]:
        """
        Yield mutable working objects after each accepted step, or only at checkpoints.
        """
        Auditor.require_integration_inputs(stepper, interval, state, snapshots=False)
        if checkpoints is not None:
            return self.mutating_trajectory_checkpoints(stepper, interval, state, checkpoints)
        return self.redirect_mutating_trajectory(stepper, interval, state)

    def snapshot(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
    ) -> tuple[IntervalLike, State]:
        return interval.copy(), stepper.snapshot_state(state)

    def stable_trajectory_checkpoints(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        copy_interval = interval.copy
        snapshot_state = stepper.snapshot_state
        for checkpoint in self.mutating_trajectory_checkpoints(stepper, interval, state, checkpoints):
            del checkpoint
            yield copy_interval(), snapshot_state(state)

    def mutating_trajectory_checkpoints(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        targets = self.checkpoint_targets(interval, checkpoints)
        mutating_impl = self.redirect_mutating_trajectory
        same_time = self.same_time
        original_stop = interval.stop
        last_positive_step = interval.step if self.advances_time(interval.present, interval.step) else None
        try:
            for index, target in enumerate(targets):
                interval.stop = target
                remaining = target - interval.present
                if interval.step <= 0.0 or not self.advances_time(interval.present, min(interval.step, remaining)):
                    interval.step = min(last_positive_step, remaining) if last_positive_step is not None else remaining
                for checkpoint in mutating_impl(stepper, interval, state):
                    del checkpoint
                    pass
                if not same_time(interval.present, target):
                    raise RuntimeError(
                        "Integration did not land on the requested checkpoint. "
                        "Use IntegratorStepper(...) or a checkpoint-aware stepper that clamps to interval.stop."
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

    def mutating_trajectory_fast(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        step = stepper
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            step(interval, state)
            yield interval, state

    def mutating_trajectory_safe(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        step = stepper
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            previous_present = interval.present
            step(interval, state)
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

    def stable_trajectory_fast(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        step = stepper
        copy_interval = interval.copy
        snapshot_state = stepper.snapshot_state
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            step(interval, state)
            yield copy_interval(), snapshot_state(state)

    def stable_trajectory_safe(
        self,
        stepper: IntegratorStepper,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        step = stepper
        copy_interval = interval.copy
        snapshot_state = stepper.snapshot_state
        same_time = self.same_time
        while interval.present < interval.stop and not same_time(interval.present, interval.stop):
            if not self.advances_time(interval.present, interval.step):
                interval.step = interval.stop - interval.present
            previous_present = interval.present
            step(interval, state)
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





