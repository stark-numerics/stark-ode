from __future__ import annotations

"""Compare derivative declaration styles on the same scalar problem."""

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods.schemes import SchemeCashKarp


FRAME = Frame({"y": {"translation": "dy", "shape": (1,)}})
INITIAL = {"y": np.array([2.0])}
INTERVAL = Interval(present=0.0, step=0.1, stop=0.2)


@DerivativeStyle.in_place
def in_place_rhs(t: float, state, out) -> None:
    del t
    out.dy[:] = -0.5 * state.y


@DerivativeStyle.returning
def returning_rhs(t: float, state):
    del t
    return {"dy": -0.5 * state.y}


@DerivativeStyle.kernel_returning(state=("y",), translation=("dy",))
def kernel_returning_rhs(y):
    return -0.5 * y


def final_value(derivative) -> float:
    system = System(derivative=derivative, frame=FRAME)
    ivp = system.ivp(
        initial={"y": INITIAL["y"].copy()},
        interval=INTERVAL,
        method=Method(scheme=SchemeCashKarp),
        engine=EngineNumpy,
        configuration=Configuration(check_progress=False),
    )
    return float(ivp.final_result().state.y[0])


print("Derivative styles")
for name, derivative in (
    ("in-place", in_place_rhs),
    ("returning", returning_rhs),
    ("kernel returning", kernel_returning_rhs),
):
    print(f"{name:16s}: y(0.2) = {final_value(derivative):.6f}")
