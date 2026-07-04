"""Use the JAX engine with a return-style derivative.

JAX arrays are immutable, so this example uses DerivativeStyle.accepts_instant_returns rather
than an in-place derivative. The solver control flow remains ordinary STARK
control flow; this example demonstrates backend storage and arithmetic, not a
fully fused JIT solve.
"""

from __future__ import annotations

try:
    import jax.numpy as jnp
    from stark.engines import EngineJax
except ImportError as exc:  # pragma: no cover - optional dependency example
    print(f"JAX backend example skipped: {exc}")
else:
    from stark import DerivativeStyle, Frame, Interval, Method, System
    from stark.methods import SchemeCashKarp

    @DerivativeStyle.accepts_instant_returns
    def decay(t: float, state):
        del t
        return {"dy": -0.5 * state.y}

    if __name__ == "__main__":
        frame = Frame.scalar("y", translation="dy")
        system = System(derivative=decay, frame=frame)
        ivp = system.ivp(
            initial={"y": jnp.array([2.0], dtype=jnp.float32)},
            interval=Interval(present=0.0, step=0.1, stop=0.3),
            method=Method(SchemeCashKarp),
            engine=lambda requested_frame: EngineJax(requested_frame, dtype=jnp.float32),
        )

        print("JAX return-style derivative")
        for interval, state in ivp.stable_trajectory():
            print(f"t={interval.present:.1f}, y={float(state.y[0]):.6f}")
