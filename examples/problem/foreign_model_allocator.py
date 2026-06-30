"""Connect an existing object model through a custom allocator.

Use `Frame` when your state is just named scalar or array data, even when the
names are nested. This example is for the other case: a model already has its
own objects, constructors, invariants, or storage conventions, and STARK should
adapt to that model instead of replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import cast

from stark import Interval
from stark.core import Integrator, IntegratorStepper
from stark.core.contracts import DerivativeLike
from stark.methods import SchemeRK4


@dataclass(slots=True)
class PendulumState:
    angle: float
    angular_velocity: float
    label: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("PendulumState.label records the owning model.")


@dataclass(slots=True)
class PendulumDelta:
    angle: float = 0.0
    angular_velocity: float = 0.0

    def __call__(self, origin: PendulumState, result: PendulumState) -> None:
        result.angle = origin.angle + self.angle
        result.angular_velocity = origin.angular_velocity + self.angular_velocity
        result.label = origin.label

    def norm(self) -> float:
        return sqrt(self.angle * self.angle + self.angular_velocity * self.angular_velocity)

    def __add__(self, other: PendulumDelta) -> PendulumDelta:
        return PendulumDelta(
            self.angle + other.angle,
            self.angular_velocity + other.angular_velocity,
        )

    def __rmul__(self, scalar: float) -> PendulumDelta:
        return PendulumDelta(
            scalar * self.angle,
            scalar * self.angular_velocity,
        )


class PendulumAllocator:
    """Allocate and copy the user's foreign state and translation objects."""

    def allocate_state(self) -> PendulumState:
        return PendulumState(angle=0.0, angular_velocity=0.0, label="foreign-pendulum")

    def copy_state(self, source: PendulumState, out: PendulumState) -> None:
        out.angle = source.angle
        out.angular_velocity = source.angular_velocity
        out.label = source.label

    def allocate_translation(self) -> PendulumDelta:
        return PendulumDelta()


def pendulum(_interval: Interval, state: PendulumState, out: PendulumDelta) -> None:
    out.angle = state.angular_velocity
    out.angular_velocity = -state.angle


if __name__ == "__main__":
    allocator = PendulumAllocator()
    state = PendulumState(angle=1.0, angular_velocity=0.0, label="foreign-pendulum")
    interval = Interval(present=0.0, step=0.1, stop=0.5)

    scheme = SchemeRK4(cast(DerivativeLike, pendulum), allocator)
    stepper = IntegratorStepper(scheme)

    print("Foreign model through a custom allocator")
    print(f"t=0.0, angle={state.angle:.6f}, velocity={state.angular_velocity:.6f}")

    for step_interval, step_state in Integrator().mutating_trajectory(stepper, interval, state):
        print(
            f"t={step_interval.present:.1f}, "
            f"angle={step_state.angle:.6f}, "
            f"velocity={step_state.angular_velocity:.6f}"
        )
