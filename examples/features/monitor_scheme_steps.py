from __future__ import annotations

"""Inspect the accepted steps recorded by scheme monitoring.

`Monitor` is the top-level object a user passes to an integration run. The
scheme itself only receives `monitor.scheme`, a narrow recording surface for
accepted scheme steps. This keeps monitored schemes lightly coupled to the
monitor implementation while still giving users concrete records to inspect.

At this stage of the monitor refactor, scheme monitoring records two distinct
kinds of evidence:

* `monitor.scheme.fixed_steps` for fixed-step schemes
* `monitor.scheme.adaptive_steps` for adaptive schemes

The records are intentionally about accepted steps. They are useful for
understanding what happened during one observed run; comparison timing should
still come from unmonitored runs.
"""

import numpy as np

from stark import Interval, IntegratorStepper
from stark.interface import StarkIVP
from stark.monitor import Monitor
from stark.schemes import SchemeCashKarp, SchemeRK4
from stark.core import Configuration

def oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


def build_problem(scheme, monitor: Monitor) -> object:
    build = StarkIVP(
        derivative=oscillator_rhs,
        initial=np.array([1.0, 0.0]),
        interval=Interval(present=0.0, step=0.2, stop=1.0),
        configuration=Configuration(),
        scheme=scheme,
    ).build()
    build.scheme = scheme(build.derivative, build.allocator, monitor=monitor.scheme)
    build.stepper = IntegratorStepper(build.scheme)
    return build


def run_with_monitor(scheme) -> Monitor:
    monitor = Monitor()
    build = build_problem(scheme, monitor)

    list(
        build.integrator.live(
            build.stepper,
            build.interval,
            build.initial,
        )
    )

    return monitor


fixed_monitor = run_with_monitor(SchemeRK4)
adaptive_monitor = run_with_monitor(SchemeCashKarp)

fixed_steps = fixed_monitor.scheme.fixed_steps
adaptive_steps = adaptive_monitor.scheme.adaptive_steps

print("Fixed-step scheme evidence")
print(f"scheme:         {fixed_steps[0].scheme}")
print(f"accepted steps: {len(fixed_steps)}")
print(f"first step:     {fixed_steps[0].t_start:.3f} -> {fixed_steps[0].t_end:.3f}")
print(f"last step:      {fixed_steps[-1].t_start:.3f} -> {fixed_steps[-1].t_end:.3f}")
print()

error_ratios = [step.error_ratio for step in adaptive_steps]
rejections = sum(step.rejection_count for step in adaptive_steps)
accepted_dt = [step.accepted_dt for step in adaptive_steps]

print("Adaptive scheme evidence")
print(f"scheme:          {adaptive_steps[0].scheme}")
print(f"accepted steps:  {len(adaptive_steps)}")
print(f"rejections:      {rejections}")
print(f"smallest dt:     {min(accepted_dt):.6g}")
print(f"largest dt:      {max(accepted_dt):.6g}")
print(f"max error ratio: {max(error_ratios):.6g}")
