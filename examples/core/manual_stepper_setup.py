"""Assemble a solver manually from an engine-owned state.

`System` is the high-level interface for solving an initial value problem.
It is the right starting point when you want to describe an ODE problem and let
STARK build the usual integration workflow.

Sometimes it is useful to work one level lower:

* when writing examples for custom schemes
* when reusing the same scheme object across several short runs
* when inspecting or changing the interval between accepted steps
* when demonstrating how states, translations, allocators, schemes, steppers,
  and integrators fit together

This example starts from a `Frame`, asks an engine for matching solver
objects, and then assembles the stepper explicitly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stark import Frame, Interval
from stark.core import Integrator, IntegratorStepper
from stark.engines import EngineNumpy
from stark.methods import SchemeRK4


def growth(
    interval: Interval,
    state: Any,
    out: Any,
) -> None:
    """Derivative for y' = y."""

    del interval
    out.dy[0] = state.y[0]


if __name__ == "__main__":
    frame = Frame.scalar("y", translation="dy")
    engine = EngineNumpy(frame)
    allocator = engine.allocator
    basis = engine.translation_basis()

    # Direct translations are useful in lower-level code that wants to express
    # displacements explicitly. The solver itself will allocate most of its
    # translations through the allocator.
    state: Any = allocator.allocate_state()
    state.y[...] = np.array([1.0])
    displacement: Any = allocator.allocate_translation()
    basis.synthesize([0.25], displacement)
    displaced_state: Any = allocator.allocate_state()
    displacement(state, displaced_state)

    print("An engine translation represents a displacement:")
    print(f"  initial state:      {state.y}")
    print(f"  after displacement: {displaced_state.y}")
    print()

    # Manual solver assembly. System would normally hide this wiring.
    scheme = SchemeRK4(growth, allocator)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)

    print("Manual stepping with RK4:")
    print(f"  t=0.0, y={state.y[0]:.6f}")

    for step_interval, step_state in Integrator().mutating_trajectory(stepper, interval, state):
        print(f"  t={step_interval.present:.1f}, y={step_state.y[0]:.6f}")
