from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from numba import njit, typeof
except ImportError:  # pragma: no cover - optional accelerator
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def jit_if_you_can(function: Callable[..., Any] | None = None, /, **kwargs):
    """
    Apply `numba.njit` when Numba is available and otherwise leave the function unchanged.

    STARK mostly uses this for tight numeric kernels, so caching is enabled by
    default and more opinionated flags remain opt-in.
    """
    options = {"cache": True, **kwargs}

    def decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        return njit(**options)(target) if NUMBA_AVAILABLE else target

    if function is None:
        return decorate
    return decorate(function)


def compile_if_you_can(function: Callable[..., Any], *signatures: Any) -> Callable[..., Any]:
    """
    Eagerly compile a jitted function from example argument tuples when possible.

    Each entry after `function` is treated as one example call. For example:

        compile_if_you_can(kernel, (out, x), (out2, x2))

    The examples are used only to infer Numba types. The kernel is not executed.
    This keeps the warm-up API close to the way STARK actually uses jitted
    kernels: configured workers often know concrete scratch arrays long before
    the first timed call, but they should not have to perform dummy work just
    to populate the JIT cache.
    """
    if not NUMBA_AVAILABLE or not signatures or not callable(function) or not hasattr(function, "compile"):
        return function

    for signature in signatures:
        arguments = signature if isinstance(signature, tuple) else (signature,)
        try:
            function.compile(tuple(typeof(argument) for argument in arguments))
        except Exception:
            continue
    return function


__all__ = ["NUMBA_AVAILABLE", "compile_if_you_can", "jit_if_you_can"]
