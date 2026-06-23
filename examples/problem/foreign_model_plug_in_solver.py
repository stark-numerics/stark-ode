"""Use STARK as a solver for an existing object model.

This is the pattern for a model that already has useful Python objects and a
home-grown time stepper. Keep the model, write a small translation and
allocator adapter, then let a STARK scheme drive the same state objects.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt

from stark import Configuration, Integrator, Interval, IntegratorStepper, Tolerance
from stark.methods import SchemeDormandPrince


@dataclass(slots=True)
class OscillatorState:
    position: float
    velocity: float
    label: str


@dataclass(slots=True)
class OscillatorDelta:
    position: float = 0.0
    velocity: float = 0.0

    def __call__(self, origin: OscillatorState, result: OscillatorState) -> None:
        result.position = origin.position + self.position
        result.velocity = origin.velocity + self.velocity
        result.label = origin.label

    def norm(self) -> float:
        return sqrt(self.position * self.position + self.velocity * self.velocity)

    def __add__(self, other: OscillatorDelta) -> OscillatorDelta:
        return OscillatorDelta(
            self.position + other.position,
            self.velocity + other.velocity,
        )

    def __rmul__(self, scalar: float) -> OscillatorDelta:
        return OscillatorDelta(scalar * self.position, scalar * self.velocity)


class OscillatorModel:
    """User-side model with its own state shape and existing Euler stepper."""

    def derivative(self, _time: float, state: OscillatorState, out: OscillatorDelta) -> None:
        out.position = state.velocity
        out.velocity = -state.position

    def euler_step(self, state: OscillatorState, dt: float) -> None:
        delta = OscillatorDelta()
        self.derivative(0.0, state, delta)
        state.position += dt * delta.position
        state.velocity += dt * delta.velocity


class OscillatorAllocator:
    """Teach STARK how to allocate and copy the user's objects."""

    def allocate_state(self) -> OscillatorState:
        return OscillatorState(position=0.0, velocity=0.0, label="foreign-oscillator")

    def copy_state(self, source: OscillatorState, out: OscillatorState) -> None:
        out.position = source.position
        out.velocity = source.velocity
        out.label = source.label

    def allocate_translation(self) -> OscillatorDelta:
        return OscillatorDelta()


def oscillator_rhs(interval: Interval, state: OscillatorState, out: OscillatorDelta) -> None:
    model.derivative(interval.present, state, out)


model = OscillatorModel()


if __name__ == "__main__":
    initial = OscillatorState(position=1.0, velocity=0.0, label="foreign-oscillator")

    euler_state = deepcopy(initial)
    for _ in range(10):
        model.euler_step(euler_state, 0.1)

    allocator = OscillatorAllocator()
    stark_state = deepcopy(initial)
    interval = Interval(present=0.0, step=0.2, stop=1.0)
    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
    )
    scheme = SchemeDormandPrince(oscillator_rhs, allocator)
    stepper = IntegratorStepper(scheme)
    integrator = Integrator(configuration=configuration)

    print("Plug an existing object model into STARK")
    print(f"Euler after t=1: position={euler_state.position:.6f}, velocity={euler_state.velocity:.6f}")
    print()
    print("STARK adaptive checkpoints")
    print(f"t=0.00, position={stark_state.position:.6f}, velocity={stark_state.velocity:.6f}")

    for checkpoint_interval, checkpoint_state in integrator.mutating_trajectory(
        stepper,
        interval,
        stark_state,
        checkpoints=5,
    ):
        print(
            f"t={checkpoint_interval.present:.2f}, "
            f"position={checkpoint_state.position:.6f}, "
            f"velocity={checkpoint_state.velocity:.6f}"
        )
