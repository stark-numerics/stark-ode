"""Use the high-level interface with the native Python engine."""

from __future__ import annotations

from array import array

from stark import Frame, Interval, Method, System
from stark.engines import EngineNative
from stark.methods import SchemeCashKarp


def exponential_decay(t, state, out):
    del t
    out.dy[0] = -0.5 * state.y[0]


def build_ivp():
    system = System(
        derivative=exponential_decay,
        frame=Frame.scalar("y", translation="dy"),
    )
    return system.ivp(
        initial={"y": array("d", [2.0])},
        interval=Interval(present=0.0, step=0.1, stop=2.0),
        method=Method(SchemeCashKarp),
        engine=EngineNative,
    )


def main() -> None:
    for interval, state in build_ivp().integrate():
        print(f"{interval.present:.3f}", state.y[0])


if __name__ == "__main__":
    main()
