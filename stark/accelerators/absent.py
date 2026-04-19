from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stark.accelerators.common import BuiltinAccelerator
from stark.contracts.acceleration import CompiledCallable


class AcceleratorAbsent(BuiltinAccelerator):
    """Trivial accelerator that leaves every callable untouched."""

    name = "none"

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        del kwargs

        def decorate_function(target: CompiledCallable) -> CompiledCallable:
            return target

        if function is None:
            return decorate_function
        return decorate_function(function)


__all__ = ["AcceleratorAbsent"]
