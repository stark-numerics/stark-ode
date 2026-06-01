from __future__ import annotations

"""Record scheme, resolvent, and inverter evidence with one Monitor.

Monitoring is opt-in at each numerical layer. The top-level `Monitor` owns
three narrow recording surfaces:

* `monitor.scheme` records accepted scheme steps;
* `monitor.resolvent` records nonlinear solve attempts;
* `monitor.inverter` records linear inverse-action attempts.

The marcher and integrator do not receive the monitor. Each worker receives
only the surface it knows how to write to.
"""

from dataclasses import dataclass

from stark import Executor, Integrator, Interval, Marcher, Monitor
from stark.block import Block
from stark.block.operator import BlockOperatorDiagonal
from stark.contracts import BlockLike, BlockOperatorLike
from stark.inverters.relaxation import InverterRelaxationRichardson
from stark.inverters.support import InverterBudget, InverterTolerance
from stark.resolvents import ResolventPicard
from stark.schemes import SchemeEuler
from stark.schemes.requests.resolvent import SchemeResolventRequest


@dataclass(slots=True)
class ScalarState:
    value: float = 0.0


@dataclass(slots=True)
class ScalarTranslation:
    value: float = 0.0

    def __call__(self, origin: ScalarState, result: ScalarState) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "ScalarTranslation") -> "ScalarTranslation":
        return ScalarTranslation(self.value + other.value)

    def __rmul__(self, scalar: float) -> "ScalarTranslation":
        return ScalarTranslation(scalar * self.value)


class ScalarAllocator:
    def allocate_state(self) -> ScalarState:
        return ScalarState()

    def copy_state(self, source: ScalarState, out: ScalarState) -> None:
        out.value = source.value

    def allocate_translation(self) -> ScalarTranslation:
        return ScalarTranslation()


@dataclass(slots=True)
class ExampleInverterRequest:
    operator: BlockOperatorLike[ScalarTranslation]
    residual: BlockLike[ScalarTranslation]


def constant_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 1.0


def zero_rhs(
    interval: Interval,
    state: ScalarState,
    out: ScalarTranslation,
) -> None:
    del interval, state
    out.value = 0.0


def scale_by_two(source: ScalarTranslation, target: ScalarTranslation) -> None:
    target.value = 2.0 * source.value


def record_scheme_level(monitor: Monitor) -> None:
    allocator = ScalarAllocator()
    scheme = SchemeEuler(constant_rhs, allocator, monitor=monitor.scheme)
    marcher = Marcher(scheme, Executor())
    interval = Interval(present=0.0, step=0.1, stop=0.3)
    state = ScalarState()

    list(Integrator().live(marcher, interval, state))


def record_resolvent_level(monitor: Monitor) -> None:
    allocator = ScalarAllocator()
    resolvent = ResolventPicard(allocator)
    resolvent.assign_monitor(monitor.resolvent)

    problem = SchemeResolventRequest(
        derivative=zero_rhs,
        interval=Interval(present=0.0, step=0.1, stop=1.0),
        origin=ScalarState(),
        rhs=None,
        alpha=0.1,
    )
    out = Block([ScalarTranslation()])

    resolvent(problem, out)


def record_inverter_level(monitor: Monitor) -> None:
    request = ExampleInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([ScalarTranslation(6.0)]),
    )
    output = Block([ScalarTranslation(0.0)])
    inverter = InverterRelaxationRichardson[ScalarTranslation](
        damping=0.5,
        tolerance=InverterTolerance(atol=1.0e-12, rtol=0.0),
        budget=InverterBudget(maximum_steps=4),
        monitor=monitor.inverter,
    )

    inverter(request, output)


def main() -> None:
    monitor = Monitor()

    record_scheme_level(monitor)
    record_resolvent_level(monitor)
    record_inverter_level(monitor)

    scheme = monitor.scheme.summary()
    resolvent = monitor.resolvent.summary()
    inverter = monitor.inverter.summary()

    print("Monitoring levels")
    print("=================")
    print(f"scheme records:    {scheme.step_count}")
    print(f"fixed steps:       {scheme.fixed_step_count}")
    print(f"resolvent solves:  {resolvent.solve_count}")
    print(f"inverter solves:   {inverter.solve_count}")
    print()

    print("Scheme records")
    for step in monitor.scheme.fixed_steps:
        print(step)
    for step in monitor.scheme.adaptive_steps:
        print(step)
    print()

    print("Resolvent records")
    for solve in monitor.resolvent.solves:
        print(solve)
    print()

    print("Inverter records")
    for solve in monitor.inverter.solves:
        print(solve)


if __name__ == "__main__":
    main()
