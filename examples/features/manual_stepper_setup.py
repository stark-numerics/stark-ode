"""Assemble a solver manually from an initial vector state.

`StarkSystem` is the high-level interface for solving an initial value problem.
It is the right starting point when you want to describe an ODE problem and let
STARK build the usual integration workflow.

Sometimes it is useful to work one level lower:

* when writing examples for custom schemes
* when reusing the same scheme object across several short runs
* when inspecting or changing the interval between accepted steps
* when demonstrating how states, translations, allocators, schemes, steppers,
  and integrators fit together

This example starts from one `StarkVector` state, derives matching solver
objects from it, and then assembles the stepper explicitly.
"""

from __future__ import annotations

from stark import Integrator, Interval, IntegratorStepper
from stark.carriers import CarrierNative
from stark.interface.vector import StarkVector, StarkVectorTranslation
from stark.schemes.explicit.fixed.rk4 import SchemeRK4


def growth(
    interval: Interval,
    state: StarkVector,
    out: StarkVectorTranslation,
) -> None:
    """Derivative for y' = y."""

    del interval
    out.value = [state.value[0]]


def main() -> None:
    carrier = CarrierNative([1.0])
    state = StarkVector([1.0], carrier)

    # The initial state is the anchor for matching solver objects.
    allocator = state.allocator()

    # Direct translations are useful in lower-level code that wants to express
    # displacements explicitly. The solver itself will allocate most of its
    # translations through the allocator.
    displacement = state.translation([0.25])
    displaced_state = allocator.allocate_state()
    displacement(state, displaced_state)

    print("A StarkVector translation represents a displacement:")
    print(f"  initial state:      {state.value}")
    print(f"  after displacement: {displaced_state.value}")
    print()

    # Manual solver assembly. StarkSystem would normally hide this wiring.
    scheme = SchemeRK4(growth, allocator)
    stepper = IntegratorStepper(scheme)
    interval = Interval(present=0.0, step=0.1, stop=0.3)

    print("Manual stepping with RK4:")
    print(f"  t=0.0, y={state.value[0]:.6f}")

    for step_interval, step_state in Integrator().mutating_trajectory(stepper, interval, state):
        print(f"  t={step_interval.present:.1f}, y={step_state.value[0]:.6f}")


if __name__ == "__main__":
    main()
