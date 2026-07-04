"""Run monitored observation separately from unmonitored timing.

Monitoring is diagnostic instrumentation: it records events, allocates summary
objects, and deliberately adds work. This script shows the recommended pattern:
first run with a `Monitor` to understand behaviour, then run the same problem
without monitoring when measuring solver speed.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from stark import DynamicsStyle, Frame, Interval, Method, Monitor, System
from stark.engines import EngineNumpy
from stark.methods import SchemeEuler


@DynamicsStyle.accepts_instant_writes
def constant(t: float, state, out) -> None:
    del t, state
    out.dx[:] = 1.0


SYSTEM = System(dynamics=constant, frame=Frame.scalar("x", translation="dx"))


def run_once(*, monitor: Monitor | None = None) -> int:
    method = Method(
        SchemeEuler,
        scheme_options={} if monitor is None else {"monitor": monitor.scheme},
    )
    ivp = SYSTEM.ivp(
        initial={"x": np.array([0.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.25),
        method=method,
        engine=EngineNumpy,
    )
    return ivp.final_result().steps


if __name__ == "__main__":
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
