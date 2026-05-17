from __future__ import annotations

"""Inspect adaptive-step behaviour with a monitor."""

import numpy as np

from stark import Executor, Interval
from stark.interface import StarkIVP
from stark.monitor import Monitor


def oscillator_rhs(t: float, y: np.ndarray) -> np.ndarray:
    del t
    return np.array([y[1], -y[0]])


monitor = Monitor()
build = StarkIVP(
    derivative=oscillator_rhs,
    initial=np.array([1.0, 0.0]),
    interval=Interval(present=0.0, step=0.05, stop=1.0),
    executor=Executor(),
).build()

list(
    build.integrator.live_monitored(
        build.marcher,
        build.interval,
        build.initial,
        monitor,
    )
)

error_ratios = [step.error_ratio for step in monitor.steps]
rejections = sum(step.rejection_count for step in monitor.steps)

print("Adaptive monitor summary")
print(f"accepted steps:  {len(monitor.steps)}")
print(f"rejections:      {rejections}")
print(f"max error ratio: {max(error_ratios):.4g}")
