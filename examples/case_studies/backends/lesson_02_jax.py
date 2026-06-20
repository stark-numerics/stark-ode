"""Lesson 2: use the JAX backend with a return-style derivative."""

from __future__ import annotations

# JAX changes the derivative style.  JAX arrays are immutable, so the example
# does not write into an output array.  Instead it returns the new translation
# field value and lets DerivativeStyle.kernel_returning adapt that to STARK's
# derivative contract.
#
# This lesson is about backend syntax and backend expectations, not about
# promising whole-solver JIT compilation.  STARK can use JAX-backed arrays and
# generated JAX-shaped Algebraist kernels, while adaptive solver control still
# happens in Python.
#
# In a source checkout, run from the ``stark-ode`` directory with:
#
#     python -m examples.case_studies.backends.lesson_02_jax

import numpy as np

from stark import Configuration, DerivativeStyle, Frame, Interval, Method, System
from stark.methods.schemes import SchemeCashKarp

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None


DIFFUSION = 0.01
REACTION = 0.25
SIZE = 256


def _kernel_returning_available() -> bool:
    return callable(getattr(DerivativeStyle, "kernel_returning", None))


def initial_jax(size: int = SIZE):
    if jnp is None:
        raise RuntimeError("JAX is not installed.")
    x = jnp.linspace(0.0, 1.0, size, endpoint=False)
    return 0.55 + 0.15 * jnp.sin(2.0 * jnp.pi * x) + 0.05 * jnp.cos(6.0 * jnp.pi * x)


def build_jax_derivative():
    if jnp is None:
        raise RuntimeError("JAX is not installed.")
    kernel_returning = getattr(DerivativeStyle, "kernel_returning", None)
    if not callable(kernel_returning):
        raise RuntimeError("This lesson requires DerivativeStyle.kernel_returning.")

    @kernel_returning(state=("u",), translation=("du",), parameters=(DIFFUSION, REACTION))
    def rhs_jax(u, diffusion: float, reaction: float):
        """Return-style JAX derivative.

        The expression is written as ordinary JAX array code.  There is no
        ``du[...] = ...`` assignment because that would be the wrong shape for
        immutable JAX arrays.
        """

        laplacian = jnp.roll(u, -1) - 2.0 * u + jnp.roll(u, 1)
        return diffusion * laplacian + reaction * u * (1.0 - u)

    return rhs_jax


def build_jax_ivp(*, size: int = SIZE):
    if jnp is None:
        raise RuntimeError("JAX is not installed.")
    from stark.engines import EngineJax

    frame = Frame({"u": {"translation": "du", "shape": (size,)}})
    system = System(derivative=build_jax_derivative(), frame=frame)
    return system.ivp(
        initial={"u": initial_jax(size)},
        interval=Interval(present=0.0, step=0.005, stop=0.015),
        method=Method(scheme=SchemeCashKarp),
        engine=EngineJax,
        configuration=Configuration(check_progress=False),
    )


def synchronize_jax(value) -> None:
    # JAX work can be asynchronous.  Synchronize before reading timing results
    # or converting the final state to NumPy diagnostics.
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()


def main() -> None:
    print("Lesson 2: JAX backend")
    print("=====================")
    print("JAX uses a return-style derivative.  This makes the derivative compatible")
    print("with immutable JAX arrays and with STARK's generated JAX Algebraist path.")
    print()

    if jnp is None:
        print("JAX is not installed; skipping this lesson.")
        return
    if not _kernel_returning_available():
        print("DerivativeStyle.kernel_returning is not available; skipping this lesson.")
        return

    ivp = build_jax_ivp()
    result = ivp.final_result()
    synchronize_jax(result.state.u)
    u = np.asarray(result.state.u)

    print(f"steps={result.steps}, final t={result.interval.present:.4f}")
    print(f"min={u.min():.5f}, max={u.max():.5f}, mean={u.mean():.5f}")
    print()
    print("What to notice:")
    print("- The derivative returns an array expression instead of mutating du.")
    print("- The final diagnostics should match the NumPy lesson up to dtype differences.")
    print("- JAX can be faster for large array work, but solver control is still Python.")


if __name__ == "__main__":
    main()
