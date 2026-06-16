from __future__ import annotations

"""Run a monitored observation pass separately from an unmonitored timing pass."""

from time import perf_counter

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, Monitor, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeEuler


@DerivativeStyle.in_place
def constant(t: float, state, out) -> None:
    del t, state
    out.dx[:] = 1.0


frame = Frame({"x": {"translation": "dx", "shape": (1,)}})
system = System(derivative=constant, frame=frame)


def run_once(*, monitor: Monitor | None = None) -> int:
    method = Method(
        scheme=SchemeEuler,
        scheme_options={} if monitor is None else {"monitor": monitor.scheme},
    )
    ivp = system.ivp(
        initial={"x": np.array([0.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.25),
        method=method,
        engine=EngineNumpy,
        configuration=Configuration(check_progress=False),
    )
    return ivp.final_result().steps


monitor = Monitor()
observed_steps = run_once(monitor=monitor)

started = perf_counter()
timed_steps = run_once(monitor=None)
elapsed = perf_counter() - started

print("Monitor versus timing")
print(f"monitored steps:   {observed_steps}")
print(f"monitor records:   {monitor.scheme.summary().step_count}")
print(f"unmonitored steps: {timed_steps}")
print(f"unmonitored time:  {elapsed:.6f}s")
print("Use monitors to explain a run; use monitor-free runs for timing tables.")
