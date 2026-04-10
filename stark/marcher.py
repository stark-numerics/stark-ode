from __future__ import annotations

from stark.audit import Auditor
from stark.control import Tolerance
from stark.contracts import IntervalLike, SchemeLike, State


class Marcher:
    __slots__ = ("scheme", "tolerance", "apply_delta_safety")

    def __init__(
        self,
        scheme: SchemeLike,
        tolerance: Tolerance | None = None,
        apply_delta_safety: bool = True,
    ) -> None:
        resolved_tolerance = tolerance if tolerance is not None else Tolerance()
        Auditor.require_marcher_inputs(scheme, resolved_tolerance, apply_delta_safety)
        self.scheme = scheme
        self.tolerance = resolved_tolerance
        self.apply_delta_safety = apply_delta_safety
        self.scheme.set_apply_delta_safety(apply_delta_safety)

    def __repr__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return (
            "Marcher("
            f"scheme={scheme_name!r}, "
            f"tolerance={self.tolerance!r}, "
            f"apply_delta_safety={self.apply_delta_safety!r})"
        )

    def __str__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return f"Marcher {scheme_name} with {self.tolerance}"

    def __call__(self, interval: IntervalLike, state: State) -> None:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return

        accepted_dt = self.scheme(interval, state, self.tolerance)
        interval.increment(accepted_dt)

    def snapshot_state(self, state: State) -> State:
        return self.scheme.snapshot_state(state)
