from __future__ import annotations

"""Use the CuPy engine when CuPy is installed."""

try:
    import cupy as cp
    from stark.engines import EngineCupy
except ImportError as exc:  # pragma: no cover - optional dependency example
    print(f"CuPy interface example skipped: {exc}")
else:
    from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
    from stark.methods.schemes import SchemeCashKarp

    @DerivativeStyle.in_place
    def decay(t: float, state, out) -> None:
        del t
        out.dy[:] = -0.5 * state.y

    frame = Frame({"y": {"translation": "dy", "shape": (1,)}})
    system = System(derivative=decay, frame=frame)
    ivp = system.ivp(
        initial={"y": cp.array([2.0])},
        interval=Interval(present=0.0, step=0.1, stop=0.3),
        method=Method(scheme=SchemeCashKarp),
        engine=EngineCupy,
        configuration=Configuration(check_progress=False),
    )

    print("CuPy in-place derivative")
    for interval, state in ivp.integrate():
        print(f"t={interval.present:.1f}, y={float(cp.asnumpy(state.y)[0]):.6f}")
