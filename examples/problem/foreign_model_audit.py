"""Audit a custom adapter for an existing object model.

When a model already has its own state objects, a custom allocator and
translation can connect it to STARK. `Auditor` gives that adapter a readable
contract check before you run a real integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import cast

from stark import Interval, Tolerance
from stark.core import Auditor
from stark.core.contracts import DerivativeLike
from stark.methods import SchemeRK4


@dataclass(slots=True)
class PendulumState:
    angle: float
    angular_velocity: float
    label: str


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
    def allocate_state(self) -> PendulumState:
        return PendulumState(angle=0.0, angular_velocity=0.0, label="foreign-pendulum")

    def copy_state(self, source: PendulumState, out: PendulumState) -> None:
        out.angle = source.angle
        out.angular_velocity = source.angular_velocity
        out.label = source.label

    def allocate_translation(self) -> PendulumDelta:
        return PendulumDelta()


def pendulum(interval: Interval, state: PendulumState, out: PendulumDelta) -> None:
    del interval
    out.angle = state.angular_velocity
    out.angular_velocity = -state.angle


if __name__ == "__main__":
    allocator = PendulumAllocator()
    state = PendulumState(angle=1.0, angular_velocity=0.0, label="foreign-pendulum")
    translation = allocator.allocate_translation()
    interval = Interval(present=0.0, step=0.1, stop=0.5)
    derivative = cast(DerivativeLike, pendulum)
    scheme = SchemeRK4(derivative, allocator)

    audit = Auditor(
        state=state,
        derivative=derivative,
        translation=translation,
        allocator=allocator,
        interval=interval,
        scheme=scheme,
        tolerance=Tolerance(atol=1.0e-9, rtol=1.0e-9),
    )

    print(audit)
    audit.raise_if_invalid()
