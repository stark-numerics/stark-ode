from __future__ import annotations

from stark.core.marcher import Marcher
from stark.executor.executor import Executor
from stark.monitor import Monitor


class DummyScheme:
    short_name = "Dummy"

    def __init__(self) -> None:
        self.monitor = None
        self.call_body = self.call_inline
        self.call_step = self.call_body
        self.redirect_call = self.call_step

    def __call__(self, interval, state, executor):
        return self.redirect_call(interval, state, executor)

    def call_inline(self, interval, state, executor):
        return 0.0

    def call_monitored(self, interval, state, executor):
        if self.monitor is not None:
            self.monitor.record_fixed_step(self.short_name, interval.present, 0.0)
        return self.call_body(interval, state, executor)

    def snapshot_state(self, state):
        return state

    def set_apply_delta_safety(self, enabled: bool) -> None:
        self.safety = enabled


def test_marcher_monitoring_rehooks_scheme_call_slots_without_scheme_lifecycle_methods() -> None:
    scheme = DummyScheme()
    marcher = Marcher(scheme, Executor())
    monitor = Monitor()

    marcher.assign_monitor(monitor)

    assert marcher.monitor is monitor
    assert scheme.monitor is monitor.scheme
    assert scheme.call_step == scheme.call_monitored
    assert scheme.redirect_call == scheme.call_step

    marcher.unassign_monitor()

    assert marcher.monitor is None
    assert scheme.monitor is None
    assert scheme.call_step == scheme.call_body
    assert scheme.redirect_call == scheme.call_step
