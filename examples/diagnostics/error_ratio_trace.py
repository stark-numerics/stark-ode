"""Inspect adaptive error ratios over time.

Adaptive schemes accept a trial step when its error ratio is below one. A trace
of those ratios tells you whether the controller is using the requested error
budget, stepping conservatively, or retrying rejected proposals.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Frame, Interval, Method, System, Tolerance
from stark.diagnostics.monitor import Monitor
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


def oscillator_rhs(t: float, state, out) -> None:
    del t
    out.dy[0] = state.y[1]
    out.dy[1] = -state.y[0]


if __name__ == "__main__":
    monitor = Monitor()
    system = System(
        derivative=oscillator_rhs,
        frame=Frame.vector("y", translation="dy", length=2),
    )
    ivp = system.ivp(
        initial={"y": np.array([1.0, 0.0])},
        interval=Interval(present=0.0, step=0.35, stop=2.0),
        method=Method(
            SchemeCashKarp,
            scheme_options={"monitor": monitor.scheme},
        ),
        engine=EngineNumpy,
        configuration=Configuration(
            scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
        ),
    )
    ivp.final_result()

    steps = monitor.scheme.adaptive_steps
    rejections = sum(step.rejection_count for step in steps)
    max_ratio = max(step.error_ratio for step in steps)

    print("Adaptive error-ratio trace")
    print(f"accepted steps:  {len(steps)}")
    print(f"rejected trials: {rejections}")
    print(f"max ratio:       {max_ratio:.6g}")
    print()
    print("step | interval        | accepted dt | error ratio | retries")
    print("-----+-----------------+-------------+-------------+--------")
    for index, step in enumerate(steps, start=1):
        print(
            f"{index:>4} | "
            f"{step.t_start:>6.3f}->{step.t_end:<6.3f} | "
            f"{step.accepted_dt:>11.6g} | "
            f"{step.error_ratio:>11.6g} | "
            f"{step.rejection_count:>7}"
        )

    print()
    print("Reading the trace:")
    print("- ratios below one were accepted")
    print("- ratios close to one use most of the requested local error budget")
    print("- repeated retries suggest the controller is hunting for a stable step size")
