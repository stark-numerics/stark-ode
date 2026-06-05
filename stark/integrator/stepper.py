"""Single-step orchestration for configured schemes."""

from __future__ import annotations

from stark.core.auditor import Auditor
from stark.contracts import IntervalLike, SchemeLike, State


class IntegratorStepper:
    __slots__ = ("scheme",)

    def __init__(
        self,
        scheme: SchemeLike,
    ) -> None:
        Auditor.require_stepper_inputs(scheme)
        self.scheme = scheme

    def __repr__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return f"IntegratorStepper(scheme={scheme_name!r})"

    def __str__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return f"IntegratorStepper {scheme_name}"

    def __call__(self, interval: IntervalLike, state: State) -> None:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return

        accepted_dt = self.scheme(interval, state)
        interval.increment(accepted_dt)

    def snapshot_state(self, state: State) -> State:
        return self.scheme.snapshot_state(state)







