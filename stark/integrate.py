from __future__ import annotations

from collections.abc import Iterable, Iterator

from stark.audit import Auditor
from stark.marcher import Marcher
from stark.contracts import IntervalLike, State


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

    When safety rails are enabled, both modes check that time advances after
    every accepted step and raise a helpful error if a scheme stalls.
    """

    __slots__ = ("safety_rails", "_snapshot_impl", "_live_impl")

    def __init__(self, safety_rails: bool = True) -> None:
        self.safety_rails = safety_rails
        self._snapshot_impl = self._call_snapshot_safe if safety_rails else self._call_snapshot_fast
        self._live_impl = self._call_live_safe if safety_rails else self._call_live_fast

    def __repr__(self) -> str:
        return f"Integrator(safety_rails={self.safety_rails!r})"

    def __str__(self) -> str:
        mode = "safe" if self.safety_rails else "fast"
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
            return self._call_snapshot_checkpoints(marcher, interval, state, checkpoints)
        return self._snapshot_impl(marcher, interval, state)

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
            return self._call_live_checkpoints(marcher, interval, state, checkpoints)
        return self._live_impl(marcher, interval, state)

    def _snapshot(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> tuple[IntervalLike, State]:
        return interval.copy(), self._snapshot_state(marcher, state)

    @staticmethod
    def _snapshot_state(marcher: Marcher, state: State) -> State:
        if hasattr(marcher, "snapshot_state"):
            return marcher.snapshot_state(state)
        raise TypeError(
            "Snapshot integration requires marcher.snapshot_state(state). "
            "Use Marcher(...) together with Integrator().live(...) for a zero-copy iterator."
        )

    def _call_snapshot_checkpoints(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        for _interval, _state in self._call_live_checkpoints(marcher, interval, state, checkpoints):
            yield self._snapshot(marcher, interval, state)

    def _call_live_checkpoints(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
        checkpoints: Checkpoints,
    ) -> Iterator[tuple[IntervalLike, State]]:
        targets = self._checkpoint_targets(interval, checkpoints)
        original_stop = interval.stop
        last_positive_step = interval.step if interval.step > 0.0 else None
        try:
            for index, target in enumerate(targets):
                interval.stop = target
                if interval.step <= 0.0 and last_positive_step is not None:
                    interval.step = min(last_positive_step, target - interval.present)
                for _interval, _state in self._live_impl(marcher, interval, state):
                    pass
                if not self._same_time(interval.present, target):
                    raise RuntimeError(
                        "Integration did not land on the requested checkpoint. "
                        "Use Marcher(...) or a checkpoint-aware marcher that clamps to interval.stop."
                    )
                if interval.step > 0.0:
                    last_positive_step = interval.step
                elif index + 1 < len(targets) and last_positive_step is None:
                    raise RuntimeError("Integration left no positive step size for the next checkpoint.")
                yield interval, state
        finally:
            interval.stop = original_stop

    @classmethod
    def _checkpoint_targets(cls, interval: IntervalLike, checkpoints: Checkpoints) -> tuple[float, ...]:
        start = float(interval.present)
        stop = float(interval.stop)
        if stop < start:
            raise ValueError("Interval stop must be greater than or equal to present.")
        if cls._same_time(start, stop):
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
            if cls._same_time(target, previous) or target < previous:
                raise ValueError("Checkpoints must be strictly increasing and after interval.present.")
            if target > stop and not cls._same_time(target, stop):
                raise ValueError("Checkpoints must not exceed interval.stop.")
            previous = target

        if not targets or not cls._same_time(targets[-1], stop):
            targets.append(stop)
        else:
            targets[-1] = stop
        return tuple(targets)

    @staticmethod
    def _same_time(left: float, right: float) -> bool:
        return abs(left - right) <= 1.0e-12 * max(1.0, abs(left), abs(right))

    def _call_live_fast(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        while interval.present < interval.stop:
            marcher(interval, state)
            yield interval, state

    def _call_live_safe(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        while interval.present < interval.stop:
            previous_present = interval.present
            marcher(interval, state)
            if interval.present <= previous_present:
                raise RuntimeError(
                    "Integration made no progress. "
                    "The scheme may have returned a non-positive step size. "
                    "Disable safety rails only if you intentionally want unchecked behavior."
                )
            yield interval, state

    def _call_snapshot_fast(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        while interval.present < interval.stop:
            marcher(interval, state)
            yield self._snapshot(marcher, interval, state)

    def _call_snapshot_safe(
        self,
        marcher: Marcher,
        interval: IntervalLike,
        state: State,
    ) -> Iterator[tuple[IntervalLike, State]]:
        if interval.step <= 0.0:
            raise ValueError("Interval step must be positive.")

        while interval.present < interval.stop:
            previous_present = interval.present
            marcher(interval, state)
            if interval.present <= previous_present:
                raise RuntimeError(
                    "Integration made no progress. "
                    "The scheme may have returned a non-positive step size. "
                    "Disable safety rails only if you intentionally want unchecked behavior."
                )
            yield self._snapshot(marcher, interval, state)

__all__ = ["Checkpoints", "Integrator"]
