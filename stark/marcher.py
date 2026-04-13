from __future__ import annotations

from stark.audit import Auditor
from stark.tolerance import Tolerance
from stark.scheme_support.tolerance import SchemeTolerance
from stark.contracts import IntervalLike, SchemeLike, State
from stark.safety import Safety


class Marcher:
    __slots__ = ("scheme", "tolerance", "safety")

    def __init__(
        self,
        scheme: SchemeLike,
        tolerance: Tolerance | None = None,
        safety: Safety | None = None,
    ) -> None:
        resolved_tolerance = tolerance if tolerance is not None else SchemeTolerance()
        resolved_safety = safety if safety is not None else Safety()
        Auditor.require_marcher_inputs(scheme, resolved_tolerance, resolved_safety)
        self.scheme = scheme
        self.tolerance = resolved_tolerance
        self.safety = resolved_safety
        self.scheme.set_apply_delta_safety(resolved_safety.apply_delta)

    def __repr__(self) -> str:
        scheme_name = getattr(self.scheme, "short_name", type(self.scheme).__name__)
        return (
            "Marcher("
            f"scheme={scheme_name!r}, "
            f"tolerance={self.tolerance!r}, "
            f"safety={self.safety!r})"
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

    def set_safety(self, safety: Safety) -> None:
        self.safety = safety
        self.scheme.set_apply_delta_safety(safety.apply_delta)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.set_safety(Safety(progress=self.safety.progress, block_sizes=self.safety.block_sizes, apply_delta=enabled))

