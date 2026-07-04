"""Use a linearizer for a real implicit Newton solve.

Implicit schemes solve nonlinear stage equations.  Newton-backed resolvents
need the Jacobian of the dynamics,

    J(y) v = d f(y)[v],

so STARK separates the dynamics from the linearizer.

This example solves a moderately stiff Van der Pol oscillator using:

    SchemeKvaerno3       adaptive implicit scheme
    ResolventNewton      nonlinear stage solver
    InverterDense        small dense linear correction solver
    LinearizerStyle      user-supplied Jacobian action / dense fill
"""
from __future__ import annotations

import numpy as np

from stark import (
    Configuration,
    DynamicsStyle,
    Frame,
    Interval,
    LinearizerStyle,
    Method,
    System,
    Tolerance,
)
from stark.core.block import BlockBasis
from stark.engines import EngineNumpy
from stark.methods import InverterDense, ResolventNewton, SchemeKvaerno3


MU = 12.0


@DynamicsStyle.kernel_accepts_instant_writes(state=("y",), translation=("dy",), parameters=(MU,))
def van_der_pol_rhs(t, y, dy, mu: float) -> None:
    """Van der Pol oscillator.

        y0' = y1
        y1' = mu * (1 - y0**2) * y1 - y0
    """

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
    """Jacobian action J(y) source -> out.

    For

        f0 = y1
        f1 = mu * (1 - y0**2) * y1 - y0

    the Jacobian is

        [ 0                         1              ]
        [ -2*mu*y0*y1 - 1           mu*(1-y0**2)   ]
    """

    del t
    y0 = y[0]
    y1 = y[1]
    v0 = source_dy[0]
    v1 = source_dy[1]

    out_dy[0] = v1
    out_dy[1] = (-2.0 * mu * y0 * y1 - 1.0) * v0 + mu * (1.0 - y0 * y0) * v1


@LinearizerStyle.dense(state=("y",), parameters=(MU,))
def van_der_pol_jacobian_dense(
    y,
    matrix,
    row_offset: int,
    column_offset: int,
    stride: int,
    mu: float,
) -> None:
    """Dense row-major fill of the same Jacobian.

    Dense inverters use this to materialise the local Newton correction matrix.
    Krylov-style inverters can use only the apply kernel above.
    """

    y0 = y[0]
    y1 = y[1]

    matrix[(row_offset + 0) * stride + column_offset + 0] = 0.0
    matrix[(row_offset + 0) * stride + column_offset + 1] = 1.0

    matrix[(row_offset + 1) * stride + column_offset + 0] = -2.0 * mu * y0 * y1 - 1.0
    matrix[(row_offset + 1) * stride + column_offset + 1] = mu * (1.0 - y0 * y0)


linearizer = LinearizerStyle.operator(
    apply=van_der_pol_jacobian_apply,
    dense=van_der_pol_jacobian_dense,
)


if __name__ == "__main__":
    frame = Frame.vector("y", translation="dy", length=2)
    system = System(
        dynamics=van_der_pol_rhs,
        linearizer=linearizer,
        frame=frame,
    )
    engine = EngineNumpy(frame)

    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=12,
    )

    prepared_linearizer = system.prepare_linearizer(engine)
    assert prepared_linearizer is not None

    inverter = InverterDense(
        basis=BlockBasis([engine.translation_basis()]),
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
        interval=Interval(present=0.0, step=0.02, stop=0.20),
        method=Method(SchemeKvaerno3, resolvent=resolvent),
        engine=lambda frame: engine,
        configuration=configuration,
    )

    print("LinearizerStyle.operator")
    print("Van der Pol oscillator solved with Newton + dense linear corrections.")
    print("The linearizer supplies J(y)v and a dense fill for the same Jacobian.")
    print()

    for interval, state in ivp.stable_trajectory(checkpoints=4):
        y0, y1 = state.y
        print(f"t={interval.present:.3f}, y0={y0:.8f}, y1={y1:.8f}")
