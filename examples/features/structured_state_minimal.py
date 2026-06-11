from __future__ import annotations

"""Minimal custom structured state example.

Use this route when the natural model is not just one scalar or array. The
state below has named fields, while the translation carries the matching
increments used by the scheme.
"""

from dataclasses import dataclass
from math import sqrt

from stark import Integrator, Interval, IntegratorStepper
from stark.methods.schemes import SchemeRK4


@dataclass(slots=True)
class Particle:
    position: float
    velocity: float


@dataclass(slots=True)
class ParticleDelta:
    position: float = 0.0
    velocity: float = 0.0

    def __call__(self, origin: Particle, result: Particle) -> None:
        result.position = origin.position + self.position
        result.velocity = origin.velocity + self.velocity

    def norm(self) -> float:
        return sqrt(self.position * self.position + self.velocity * self.velocity)

    def __add__(self, other: "ParticleDelta") -> "ParticleDelta":
        return ParticleDelta(
            self.position + other.position,
            self.velocity + other.velocity,
        )

    def __rmul__(self, scalar: float) -> "ParticleDelta":
        return ParticleDelta(
            scalar * self.position,
            scalar * self.velocity,
        )


class ParticleAllocator:
    def allocate_state(self) -> Particle:
        return Particle(0.0, 0.0)

    def copy_state(self, source: Particle, out: Particle) -> None:
        out.position = source.position
        out.velocity = source.velocity

    def allocate_translation(self) -> ParticleDelta:
        return ParticleDelta()


def harmonic_motion(
    interval: Interval,
    state: Particle,
    out: ParticleDelta,
) -> None:
    del interval
    out.position = state.velocity
    out.velocity = -state.position


allocator = ParticleAllocator()
scheme = SchemeRK4(harmonic_motion, allocator)
stepper = IntegratorStepper(scheme)
interval = Interval(present=0.0, step=0.1, stop=0.5)
state = Particle(position=1.0, velocity=0.0)

print("Structured particle state")
for interval, state in Integrator().mutating_trajectory(stepper, interval, state):
    print(f"t={interval.present:.1f}, x={state.position:.6f}, v={state.velocity:.6f}")

