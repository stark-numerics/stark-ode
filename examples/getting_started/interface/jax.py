from __future__ import annotations

"""Use the JAX engine with a return-style derivative.

JAX arrays are immutable, so this example uses DerivativeStyle.returning rather
than an in-place derivative. The solver control flow remains ordinary STARK
control flow; this example demonstrates backend storage and arithmetic, not a
fully fused JIT solve.
"""

try:
    import jax.numpy as jnp
    from stark.engines import EngineJax
except ImportError as exc:  # pragma: no cover - optional dependency example
    print(f"JAX interface example skipped: {exc}")
else:
    from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
    from stark.methods.schemes import SchemeCashKarp

    @DerivativeStyle.returning
    def decay(t: float, state):
        del t
        return {"dy": -0.5 * state.y}

    frame = Frame({"y": {"translation": "dy", "shape": (1,)}})
    system = System(derivative=decay, frame=frame)
    ivp = system.ivp(
        initial={"y": jnp.array([2.0], dtype=jnp.float32)},
        interval=Interval(present=0.0, step=0.1, stop=0.3),
        method=Method(scheme=SchemeCashKarp),
        engine=lambda requested_frame: EngineJax(requested_frame, dtype=jnp.float32),
        configuration=Configuration(check_progress=False),
    )

    print("JAX return-style derivative")
    for interval, state in ivp.integrate():
        print(f"t={interval.present:.1f}, y={float(state.y[0]):.6f}")
