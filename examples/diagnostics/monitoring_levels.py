"""Record scheme, resolvent, and inverter evidence with one Monitor.

Monitoring is opt-in at each numerical layer. Schemes receive
`monitor.scheme` through the method options used to build an IVP. Resolvents
and inverters receive their own monitor surfaces when they are exercised
directly.
"""

from __future__ import annotations

import numpy as np

from stark import Configuration, Frame, Interval, Method, Monitor, System, Tolerance
from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.engines import EngineNumpy
from stark.methods import InverterRelaxationRichardson, ResolventPicard, SchemeEuler
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from stark.methods.schemes.requests.resolvent import SchemeResolventRequest


FRAME = Frame.scalar("x", translation="dx")


def constant_rhs(t: float, state, out) -> None:
    del t, state
    out.dx[0] = 1.0


def zero_rhs(interval, state, out) -> None:
    del interval, state
    out.dx[:] = 0.0


def scale_by_two(source, target) -> None:
    target.dx[:] = 2.0 * source.dx


def record_scheme_level(monitor: Monitor) -> None:
    system = System(derivative=constant_rhs, frame=FRAME)
    ivp = system.ivp(
        initial={"x": np.array([0.0])},
        interval=Interval(present=0.0, step=0.1, stop=0.3),
        method=Method(SchemeEuler, scheme_options={"monitor": monitor.scheme}),
        engine=EngineNumpy,
    )

    ivp.final_result()


def record_resolvent_level(monitor: Monitor) -> None:
    engine = EngineNumpy(FRAME)
    resolvent = ResolventPicard(engine.allocator)
    resolvent.assign_monitor(monitor.resolvent)

    request = SchemeResolventRequest(
        derivative=zero_rhs,
        interval=Interval(present=0.0, step=0.1, stop=1.0),
        origin=engine.allocator.allocate_state(),
        rhs=None,
        alpha=0.1,
    )
    delta = Block([engine.allocator.allocate_translation()])

    resolvent(request, delta)


def record_inverter_level(monitor: Monitor) -> None:
    engine = EngineNumpy(FRAME)
    residual = engine.allocator.allocate_translation()
    residual.dx[0] = 6.0
    output_delta = engine.allocator.allocate_translation()

    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([residual]),
    )
    output = Block([output_delta])
    inverter = InverterRelaxationRichardson(
        damping=0.5,
        configuration=Configuration(
            inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0),
            inverter_maximum_steps=4,
        ),
        monitor=monitor.inverter,
    )

    inverter(request, output)


if __name__ == "__main__":
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
