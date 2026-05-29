from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from stark.contracts.acceleration import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorBase:
    """Shared machinery for STARK's built-in accelerator workers."""

    strict: bool = False
    values: dict[str, Any] = field(default_factory=dict, repr=False)

    name: ClassVar[str]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(strict={self.strict!r}, values={self.values!r})"

    def __str__(self) -> str:
        return self.name

    def with_updates(self, **updates: Any) -> "AcceleratorBase":
        values = dict(self.values)
        strict = self.strict
        if "strict" in updates:
            strict = updates.pop("strict")
        values.update(updates)
        return replace(self, strict=strict, values=values)

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
        raise NotImplementedError

    def compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        compiled = self._compile_examples(function, *examples)
        if self.strict and compiled is function and self.name != "none":
            raise RuntimeError(f"{self.name} backend could not compile the requested callable.")
        return compiled

    def _compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        del examples
        return function


__all__ = ["AcceleratorBase"]
