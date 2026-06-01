"""Minimal fixed-step explicit scheme example.

This example shows the public shape expected from a STARK scheme:

- `__call__(interval, state, executor) -> float`
- `snapshot_state(state)`

The example uses `StarkVector`, STARK's simple vector-space carrier. In this
case the state and the increment both live in the same mathematical vector
space, even though STARK still distinguishes the state wrapper from the
translation wrapper internally.
"""

from __future__ import annotations

from stark import Executor, Integrator, Interval, Marcher
from stark.carriers import CarrierNative
from stark.interface.vector import (
    StarkVector,
    StarkVectorTranslation,
    StarkVectorAllocator,
)


def derivative(
    interval: Interval,
    state: StarkVector,
    out: StarkVectorTranslation,
) -> None:
    del interval
    out.value[:] = state.value


class ForwardEuler:
    """Minimal custom fixed-step explicit scheme.

    STARK already includes a production `SchemeEuler`; this class exists only to
    show the public scheme contract in the smallest possible form.

    `SchemeRK4` is a richer built-in example of a fixed explicit scheme with
    reusable buffers and optional generated stage algebra.
    """

    def __init__(self) -> None:
        carrier = CarrierNative([1.0])
        self.allocator = StarkVectorAllocator(carrier)
        self.delta = self.allocator.allocate_translation()

    def __call__(
        self,
        interval: Interval,
        state: StarkVector,
        executor: Executor,
    ) -> float:
        del executor

        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return 0.0

        dt = interval.step if interval.step <= remaining else remaining

        derivative(interval, state, self.delta)

        update = dt * self.delta
        update(state, state)

        return dt

    def snapshot_state(self, state: StarkVector) -> StarkVector:
        snapshot = self.allocator.allocate_state()
        self.allocator.copy_state(state, snapshot)
        return snapshot

def main() -> None:
    scheme = ForwardEuler()
    marcher = Marcher(scheme, Executor())

    carrier = scheme.allocator.carrier
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = StarkVector([1.0], carrier)

    for snapshot_interval, snapshot_state in Integrator().live(marcher, interval, state):
        y = snapshot_state.value[0]
        print(f"t={snapshot_interval.present:.1f}, y={y:.6f}")


if __name__ == "__main__":
    main()
