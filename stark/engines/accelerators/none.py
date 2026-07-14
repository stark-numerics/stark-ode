from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, overload

from stark.core.contracts.accelerator import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorNone:
    """Trivial accelerator that leaves every callable untouched."""

    strict: bool = False

    name: ClassVar[str] = "none"

    def __str__(self) -> str:
        return self.name

    @overload
    def compile(
        self,
        function: None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[[AcceleratorTarget], AcceleratorTarget]:
        ...

    @overload
    def compile(
        self,
        function: AcceleratorTarget,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget:
        ...

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget | Callable[[AcceleratorTarget], AcceleratorTarget]:
        del label, cache, options

        def compile_function(target: AcceleratorTarget) -> AcceleratorTarget:
            return target

        if function is None:
            return compile_function
        return compile_function(function)

    def compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        del examples
        return function


__all__ = ["AcceleratorNone"]
