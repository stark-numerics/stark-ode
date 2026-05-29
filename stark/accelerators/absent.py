from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stark.accelerators.common import AcceleratorBase
from stark.contracts.acceleration import AcceleratorTarget


class AcceleratorAbsent(AcceleratorBase):
    """Trivial accelerator that leaves every callable untouched."""

    name = "none"

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        del label, cache, options

        def compile_function(target: AcceleratorTarget) -> AcceleratorTarget:
            return target

        if function is None:
            return compile_function
        return compile_function(function)


__all__ = ["AcceleratorAbsent"]
