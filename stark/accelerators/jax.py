from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from stark.accelerators.common import AcceleratorBase
from stark.contracts.acceleration import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorJax(AcceleratorBase):
    """JAX-backed accelerator for pure array functions."""

    _jax: Any = field(init=False, repr=False, default=None)

    name = "jax"

    def __init__(self, *, strict: bool = False, values: dict[str, Any] | None = None) -> None:
        try:
            import jax
        except ImportError:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("AcceleratorJax requires JAX to be installed.") from None
        self.strict = strict
        self.values = {} if values is None else dict(values)
        self._jax = jax

    def decorate(self, function: AcceleratorTarget | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        options = dict(kwargs)

        def decorate_function(target: AcceleratorTarget) -> AcceleratorTarget:
            return self._jax.jit(target, **options)

        if function is None:
            return decorate_function
        return decorate_function(function)

    def _compile_examples(self, function: AcceleratorTarget, *signatures: Any) -> AcceleratorTarget:
        if not signatures or not callable(function):
            return function

        lower = getattr(function, "lower", None)
        if callable(lower):
            for signature in signatures:
                arguments = signature if isinstance(signature, tuple) else (signature,)
                try:
                    lower(*arguments).compile()
                except Exception:
                    continue
            return function

        for signature in signatures:
            arguments = signature if isinstance(signature, tuple) else (signature,)
            try:
                function(*arguments)
            except Exception:
                continue
        return function


__all__ = ["AcceleratorJax"]
