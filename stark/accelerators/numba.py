from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from stark.accelerators.common import BuiltinAccelerator
from stark.contracts.acceleration import CompiledCallable


@dataclass(slots=True)
class AcceleratorNumba(BuiltinAccelerator):
    """Numba-backed accelerator for imperative numerical kernels."""

    cache: bool = True
    _njit: Any = field(init=False, repr=False, default=None)
    _typeof: Any = field(init=False, repr=False, default=None)

    name = "numba"

    def __init__(self, *, cache: bool = True, strict: bool = False, values: dict[str, Any] | None = None) -> None:
        try:
            from numba import njit, typeof
        except ImportError:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("AcceleratorNumba requires Numba to be installed.") from None
        self.cache = cache
        self.strict = strict
        self.values = {} if values is None else dict(values)
        self._njit = njit
        self._typeof = typeof

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        options = {"cache": self.cache, **kwargs}

        def decorate_function(target: CompiledCallable) -> CompiledCallable:
            return self._njit(**options)(target)

        if function is None:
            return decorate_function
        return decorate_function(function)

    def _compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        if not signatures or not callable(function) or not hasattr(function, "compile"):
            return function

        for signature in signatures:
            arguments = signature if isinstance(signature, tuple) else (signature,)
            try:
                function.compile(tuple(self._typeof(argument) for argument in arguments))
            except Exception:
                continue
        return function


__all__ = ["AcceleratorNumba"]
