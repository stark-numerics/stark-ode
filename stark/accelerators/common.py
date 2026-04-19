from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from stark.contracts.acceleration import (
    AccelerationRequest,
    AccelerationRole,
    CompiledCallable,
)


@dataclass(slots=True)
class BuiltinAccelerator:
    """Shared machinery for STARK's built-in accelerator workers."""

    strict: bool = False
    values: dict[str, Any] = field(default_factory=dict, repr=False)
    available: bool = field(init=False, default=True)

    name: ClassVar[str]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(available={self.available!r}, strict={self.strict!r}, values={self.values!r})"

    def __str__(self) -> str:
        state = "ready" if self.available else "unavailable"
        return f"{self.name} ({state})"

    def with_updates(self, **updates: Any) -> "BuiltinAccelerator":
        values = dict(self.values)
        strict = self.strict
        if "strict" in updates:
            strict = updates.pop("strict")
        values.update(updates)
        return replace(self, strict=strict, values=values)

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        raise NotImplementedError

    def compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        compiled = self._compile_examples(function, *signatures)
        if self.strict and compiled is function and self.name != "none":
            raise RuntimeError(f"{self.name} backend could not compile the requested callable.")
        return compiled

    def _compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        del signatures
        return function

    def resolve(self, target: Any, request: AccelerationRequest) -> Any:
        hook = getattr(target, "accelerated", None)
        if callable(hook):
            variant = hook(self, request)
            if variant is not None:
                return variant
        return target

    def resolve_derivative(self, derivative: Any) -> Any:
        return self.resolve(derivative, AccelerationRequest(AccelerationRole.DERIVATIVE))

    def resolve_linearizer(self, linearizer: Any) -> Any:
        return self.resolve(linearizer, AccelerationRequest(AccelerationRole.LINEARIZER))

    def resolve_support(self, worker: Any, *, label: str | None = None, **values: Any) -> Any:
        return self.resolve(worker, AccelerationRequest(AccelerationRole.SUPPORT, label=label, values=values))


__all__ = ["BuiltinAccelerator"]
