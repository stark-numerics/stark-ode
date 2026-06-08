from __future__ import annotations

"""Inspect the accepted steps recorded by scheme monitoring."""

import numpy as np

from stark import (
    Configuration,
    Interval,
    StarkLayout,
    StarkMethod,
    StarkSystem,
)
from stark.engines import StarkEngineNumpy
from stark.monitor import Monitor
from stark.schemes import SchemeCashKarp, SchemeRK4


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


system = StarkSystem(
    derivative=oscillator_rhs,
    layout=StarkLayout({"y": {"translation": "dy", "shape": (2,)}}),
)


def build_problem(scheme_type, monitor: Monitor):
    return system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.2, stop=1.0),
        method=StarkMethod(scheme=scheme_type, scheme_options={"monitor": monitor.scheme}),
        engine=StarkEngineNumpy,
        configuration=Configuration(check_progress=False),
    )


def run_with_monitor(scheme_type) -> Monitor:
    monitor = Monitor()
    ivp = build_problem(scheme_type, monitor)

    list(ivp.mutating_trajectory())

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
