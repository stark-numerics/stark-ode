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
    available: bool = field(init=False)
    _njit: Any = field(init=False, repr=False, default=None)
    _typeof: Any = field(init=False, repr=False, default=None)

    name = "numba"

    def __post_init__(self) -> None:
        try:
            from numba import njit, typeof
        except ImportError:  # pragma: no cover - optional dependency
            self.available = False
            self._njit = None
            self._typeof = None
        else:
            self.available = True
            self._njit = njit
            self._typeof = typeof

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        options = {"cache": self.cache, **kwargs}

        def decorate_function(target: CompiledCallable) -> CompiledCallable:
            if not self.available or self._njit is None:
                return target
            return self._njit(**options)(target)

        if function is None:
            return decorate_function
        return decorate_function(function)

    def _compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        if not self.available or not signatures or not callable(function) or not hasattr(function, "compile"):
            return function

        assert self._typeof is not None
        for signature in signatures:
            arguments = signature if isinstance(signature, tuple) else (signature,)
            try:
                function.compile(tuple(self._typeof(argument) for argument in arguments))
            except Exception:
                continue
        return function


__all__ = ["AcceleratorNumba"]
