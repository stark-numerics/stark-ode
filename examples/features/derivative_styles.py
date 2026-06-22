"""Compare derivative declaration styles on the same scalar problem."""

from __future__ import annotations

import numpy as np

from stark import DerivativeStyle, Frame, Interval, Method, System
from stark.engines import EngineNumpy
from stark.methods import SchemeCashKarp


FRAME = Frame.scalar("y", translation="dy")
INITIAL = {"y": np.array([2.0])}
INTERVAL = Interval(present=0.0, step=0.1, stop=0.2)


@DerivativeStyle.accepts_instant_writes
def in_place_rhs(t: float, state, out) -> None:
    out.dy[:] = -(0.5 + 0.1 * t) * state.y


@DerivativeStyle.accepts_instant_returns
def returning_rhs(t: float, state):
    return {"dy": -(0.5 + 0.1 * t) * state.y}


@DerivativeStyle.kernel_accepts_instant_returns(state=("y",), translation=("dy",))
def kernel_returning_rhs(t, y):
    return -(0.5 + 0.1 * t) * y


def final_value(derivative) -> float:
    system = System(derivative=derivative, frame=FRAME)
    ivp = system.ivp(
        initial={"y": INITIAL["y"].copy()},
        interval=INTERVAL,
        method=Method(SchemeCashKarp),
        engine=EngineNumpy,
    )
    return float(ivp.final_result().state.y[0])


def main() -> None:
    print("Derivative styles")
    for name, derivative in (
        ("in-place", in_place_rhs),
        ("returning", returning_rhs),
        ("kernel returning", kernel_returning_rhs),
    ):
        print(f"{name:16s}: y(0.2) = {final_value(derivative):.6f}")


if __name__ == "__main__":
    main()
