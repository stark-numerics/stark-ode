from __future__ import annotations

from typing import Any, Protocol, Self

from stark.contracts.audit_support import AuditRecorder


class IntervalLike(Protocol):
    """
    Protocol for a rolling integration interval.

    User-defined intervals are fine as long as they expose the same attributes
    and behavior as STARK's primitive `Interval`.
    """

    present: float
    step: float
    stop: float

    def copy(self) -> Self:
        ...

    def increment(self, dt: float) -> None:
        ...


class IntervalAudit:
    def __call__(self, recorder: AuditRecorder, interval: Any, *, exercise: bool = True) -> None:
        for name in ("present", "step", "stop"):
            recorder.check(hasattr(interval, name), f"Interval provides {name}.", f"Add a {name} attribute to the interval.")

        increment = getattr(interval, "increment", None)
        copy = getattr(interval, "copy", None)
        recorder.check(callable(increment), "Interval provides increment(dt).", "Add increment(dt) to advance present by the accepted step.")
        recorder.check(callable(copy), "Interval provides copy().", "Add copy() so snapshot integration can clone the interval.")

        if not exercise:
            return

        if callable(copy):
            try:
                interval_copy = copy()
            except Exception as exc:
                recorder.record_exception("Interval.copy() succeeds.", exc)
            else:
                recorder.check(True, "Interval.copy() succeeds.")
                copied_present = getattr(interval_copy, "present", None)
                copied_step = getattr(interval_copy, "step", None)
                copied_stop = getattr(interval_copy, "stop", None)
                recorder.check(
                    copied_present == getattr(interval, "present", None)
                    and copied_step == getattr(interval, "step", None)
                    and copied_stop == getattr(interval, "stop", None),
                    "Interval.copy() preserves present, step, and stop.",
                    "Make copy() return an interval snapshot with matching present, step, and stop.",
                )

                if callable(increment):
                    try:
                        before = interval_copy.present
                        interval_copy.increment(0.125)
                    except Exception as exc:
                        recorder.record_exception("Interval.increment(dt) succeeds on a copy.", exc)
                    else:
                        after = interval_copy.present
                        recorder.check(
                            after > before,
                            "Interval.increment(dt) advances present on a copy.",
                            "Make increment(dt) increase present by the accepted step size.",
                        )


__all__ = ["IntervalAudit", "IntervalLike"]









