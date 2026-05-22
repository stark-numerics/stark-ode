from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stark.accelerators.common import AcceleratorBase
from stark.contracts.acceleration import AcceleratorTarget


class AcceleratorAbsent(AcceleratorBase):
    """Trivial accelerator that leaves every callable untouched."""

    name = "none"

    def decorate(self, function: AcceleratorTarget | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        del kwargs

        def decorate_function(target: AcceleratorTarget) -> AcceleratorTarget:
            return target

        if function is None:
            return decorate_function
        return decorate_function(function)


__all__ = ["AcceleratorAbsent"]
