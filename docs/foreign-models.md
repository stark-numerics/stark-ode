# Integrate a foreign model

This page is for users whose model already has its own state objects and solver increments.

Use the high-level `System`/`Frame` path when your state can be represented as named fields. Use this page when that would be artificial.

## The idea

STARK separates:

```text
state        model configuration
translation  solver increment / tangent object
allocator    factory for states, translations, and scratch
```

A foreign model integration provides these objects directly.

## Minimal custom state example

This is a small harmonic oscillator with custom Python objects rather than `Frame`-generated state.

```python
from __future__ import annotations

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


def harmonic_motion(interval: Interval, state: Particle, out: ParticleDelta) -> None:
    del interval
    out.position = state.velocity
    out.velocity = -state.position


allocator = ParticleAllocator()
scheme = SchemeRK4(harmonic_motion, allocator)
stepper = IntegratorStepper(scheme)
interval = Interval(present=0.0, step=0.1, stop=0.5)
state = Particle(position=1.0, velocity=0.0)

for interval, state in Integrator().mutating_trajectory(stepper, interval, state):
    print(f"t={interval.present:.1f}, x={state.position:.6f}, v={state.velocity:.6f}")
```

Run the maintained version:

```powershell
python -m examples.features.structured_state_minimal
```

## What the translation must do

A translation must be able to:

```text
apply itself to a state
be scaled and combined
report a norm when adaptive schemes need one
```

For simple foreign objects, Python special methods may be enough. For high-performance foreign models, provide explicit operations through an allocator or engine-specific path.

## What the allocator must do

An allocator gives schemes and integrators fresh objects without knowing the model's constructor details.

At minimum:

```text
allocate_state
copy_state
allocate_translation
```

Implicit methods and advanced inverters may require additional operator or block allocation support.

## When to prefer `Frame`

Use `Frame` if your model can be described as named array/scalar fields. The high-level path gives STARK more structure, which enables generated Algebraist kernels and backend acceleration.

Use custom state/translation only when preserving the foreign model representation is more important than the generated high-level path.

## Next

- [Mathematical contracts](contracts_math.md) explains the formal state/translation model.
- [Extending STARK](extending.md) explains method components such as schemes and inverters.
- [Algebraist backend paths](contributing/algebraist_backends.md) explains why `Frame`-backed models can use generated kernels while foreign models may need runtime fallback.
