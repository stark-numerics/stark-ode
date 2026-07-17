from __future__ import annotations

from warnings import warn
from typing import Any, cast

from stark.core.contracts import LinearCombine


def require_allocator_linear_combine(
    allocator: Any,
    *,
    consumer: str,
) -> LinearCombine:
    """Return an allocator's linear-combine table or fail during setup."""

    try:
        linear_combine = allocator.linear_combine
    except AttributeError as exc:
        message = (
            f"{consumer} requires allocator.linear_combine kernels, but the "
            f"allocator {type(allocator).__name__!r} does not provide them. "
            "Decorate the allocator class with @Allocator.runtime, optionally "
            "add @Allocator.linear_combine(...) seed kernels, and construct "
            "that decorated allocator before constructing "
            f"{consumer}."
        )
        warn(message, RuntimeWarning, stacklevel=2)
        raise ValueError(message) from exc

    if not isinstance(linear_combine, tuple):
        message = (
            f"{consumer} requires allocator.linear_combine to be a tuple of "
            f"kernels; got {type(linear_combine).__name__!r}."
        )
        warn(message, RuntimeWarning, stacklevel=2)
        raise ValueError(message)

    return cast(LinearCombine, linear_combine)


def require_linear_combine_kernels(
    allocator: Any,
    *,
    arity: int,
    consumer: str,
) -> tuple[Any, ...]:
    """Return prepared linear-combine kernels."""

    if arity < 1:
        raise ValueError(f"{consumer} requires a positive linear-combine arity.")

    linear_combine = require_allocator_linear_combine(allocator, consumer=consumer)
    if len(linear_combine) < arity:
        message = (
            f"{consumer} requires allocator.linear_combine to provide at least "
            f"{arity} kernel(s), covering translation arities 1..{arity}; "
            f"got {len(linear_combine)}. Decorate the allocator class with "
            "@Allocator.runtime before constructing it so STARK can prepare a "
            "complete table, or provide a complete custom linear_combine tuple "
            "on the allocator."
        )
        warn(message, RuntimeWarning, stacklevel=2)
        raise ValueError(message)
    return linear_combine[:arity]


__all__ = ["require_allocator_linear_combine", "require_linear_combine_kernels"]
