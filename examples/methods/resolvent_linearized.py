"""Use a linearized resolvent for an implicit stage.

Newton resolves an implicit stage by repeatedly solving a linear correction
problem. That needs a problem linearizer and an inverter. This example keeps the
ODE scalar so the moving parts stay visible.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from stark import (
    Configuration,
    Frame,
    Interval,
    LinearizerStyle,
    Method,
    System,
    Tolerance,
)
from stark.core.block import BlockBasis
from stark.engines import EngineNumpy
from stark.methods import InverterDense, ResolventNewton, SchemeBackwardEuler


def decay_rhs(t: float, state: Any, out: Any) -> None:
    del t
    out.dy[0] = -0.5 * state.y[0]


@LinearizerStyle.kernel_accepts_instant_writes(
    state=("y",),
    source=("dy",),
    target=("dy",),
)
def decay_jacobian_apply(t, y, source_dy, out_dy) -> None:
    out_dy[0] = -0.5 * source_dy[0]


@LinearizerStyle.dense(state=("y",))
def decay_jacobian_dense(y, matrix, row_offset: int, column_offset: int, stride: int) -> None:
    matrix[row_offset * stride + column_offset] = -0.5


linearizer = LinearizerStyle.operator(
    apply=decay_jacobian_apply,
    dense=decay_jacobian_dense,
)


if __name__ == "__main__":
    frame = Frame.scalar("y", translation="dy")
    system = System(
        dynamics=decay_rhs,
        linearizer=linearizer,
        frame=frame,
    )
    engine = EngineNumpy(frame)
    configuration = Configuration(
        scheme_tolerance=Tolerance(atol=1.0e-8, rtol=1.0e-6),
        resolvent_tolerance=Tolerance(atol=1.0e-10, rtol=1.0e-8),
        resolvent_maximum_steps=8,
    )
    prepared_linearizer = system.prepare_linearizer(engine)
    assert prepared_linearizer is not None

    inverter = InverterDense(
        basis=BlockBasis([engine.translation_basis()]),
        accelerator=engine.accelerator,
    )
    resolvent = ResolventNewton(
        allocator=engine.allocator,
        linearizer=prepared_linearizer,
        inverter=inverter,
        configuration=configuration,
        accelerator=engine.accelerator,
        tableau=SchemeBackwardEuler.tableau,
    )

    ivp = system.ivp(
        initial={"y": np.array([2.0])},
        interval=Interval(present=0.0, step=0.05, stop=0.5),
        method=Method(SchemeBackwardEuler, resolvent=resolvent),
        engine=lambda frame: engine,
        configuration=configuration,
    )
    result = ivp.final_result()

    print("Linearized resolvent: Newton + dense inverter")
    print(f"accepted steps: {result.steps}")
    print(f"final t:        {result.interval.present:.6f}")
    print(f"final y:        {result.state.y[0]:.8f}")
