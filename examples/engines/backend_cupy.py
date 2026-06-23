"""Use the CuPy engine when CuPy is installed."""

from __future__ import annotations

try:
    import cupy as cp
    from stark.engines import EngineCupy
except ImportError as exc:  # pragma: no cover - optional dependency example
    print(f"CuPy backend example skipped: {exc}")
else:
    from stark import DerivativeStyle, Frame, Interval, Method, System
    from stark.methods import SchemeCashKarp

    @DerivativeStyle.accepts_instant_writes
    def decay(t: float, state, out) -> None:
        del t
        out.dy[:] = -0.5 * state.y

    if __name__ == "__main__":
        frame = Frame.scalar("y", translation="dy")
        system = System(derivative=decay, frame=frame)
        ivp = system.ivp(
            initial={"y": cp.array([2.0])},
            interval=Interval(present=0.0, step=0.1, stop=0.3),
            method=Method(SchemeCashKarp),
            engine=EngineCupy,
        )

        print("CuPy in-place derivative")
        for interval, state in ivp.integrate():
            print(f"t={interval.present:.1f}, y={float(cp.asnumpy(state.y)[0]):.6f}")
