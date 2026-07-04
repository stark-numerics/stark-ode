"""Use a matrix-free Jacobian with Newton and a Krylov inverter.

Newton needs the action of the derivative Jacobian, but a dense matrix is not
always the right representation. This example supplies only ``J(y) v`` and
lets a Krylov inverter solve the linear correction problem matrix-free.
"""

from __future__ import annotations

import numpy as np

from stark import (
    Configuration,
    DerivativeStyle,
    Frame,
    Interval,
    LinearizerStyle,
    Method,
    System,
    Tolerance,
)
from stark.engines import EngineNumpy
from stark.methods import InverterKrylovArnoldi, ResolventNewton, SchemeKvaerno3


MU = 12.0


@DerivativeStyle.kernel_accepts_instant_writes(
    state=("y",),
    translation=("dy",),
    parameters=(MU,),
)
def van_der_pol_rhs(t, y, dy, mu: float) -> None:
    y0 = y[0]
    y1 = y[1]
    dy[0] = y1
    dy[1] = mu * (1.0 - y0 * y0) * y1 - y0


@LinearizerStyle.kernel_accepts_instant_writes(
    state=("y",),
    source=("dy",),
    target=("dy",),
    parameters=(MU,),
)
def van_der_pol_jacobian_apply(t, y, source_dy, out_dy, mu: float) -> None:
    y0 = y[0]
    y1 = y[1]
    v0 = source_dy[0]
    v1 = source_dy[1]

    out_dy[0] = v1
    out_dy[1] = (-2.0 * mu * y0 * y1 - 1.0) * v0 + mu * (1.0 - y0 * y0) * v1


linearizer = LinearizerStyle.operator(apply=van_der_pol_jacobian_apply)


if __name__ == "__main__":
    frame = Frame.vector("y", translation="dy", length=2)
    system = System(
        derivative=van_der_pol_rhs,
        linearizer=linearizer,
        frame=frame,
    )
    engine = EngineNumpy(frame)
    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=12,
        inverter_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        inverter_maximum_steps=12,
    )

    prepared_linearizer = system.prepare_linearizer(engine)
    assert prepared_linearizer is not None

    inverter = InverterKrylovArnoldi(
        engine.allocator,
        engine.allocator.inner_product,
        restart=4,
        configuration=configuration,
        accelerator=engine.accelerator,
    )
    resolvent = ResolventNewton(
        engine.allocator,
        linearizer=prepared_linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=SchemeKvaerno3.tableau,
    )
    ivp = system.ivp(
        initial={"y": np.array([2.0, 0.0])},
        interval=Interval(present=0.0, step=0.02, stop=0.2),
        method=Method(SchemeKvaerno3, resolvent=resolvent),
        engine=lambda frame: engine,
        configuration=configuration,
    )

    print("Matrix-free Jacobian: Newton + Krylov")
    for interval, state in ivp.stable_trajectory(checkpoints=4):
        y0, y1 = state.y
        print(f"t={interval.present:.3f}, y0={y0:.8f}, y1={y1:.8f}")
