from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, overload

from stark.core.contracts.accelerator import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorJax:
    """JAX-backed accelerator for pure array functions."""

    strict: bool = False
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    _jax: Any = field(init=False, repr=False, default=None)

    name: ClassVar[str] = "jax"

    def __init__(self, *, strict: bool = False, options: dict[str, Any] | None = None) -> None:
        try:
            import jax
        except ImportError:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("AcceleratorJax requires JAX to be installed.") from None
        self.strict = strict
        self.options = {} if options is None else dict(options)
        self._jax = jax

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
        del label, cache
        jit_options = {**self.options, **options}

        def compile_function(target: AcceleratorTarget) -> AcceleratorTarget:
            if getattr(target, "__code__", None) is None:
                if self.strict:
                    raise RuntimeError("jax backend can only compile plain Python functions.")
                return target

            return self._jax.jit(target, **jit_options)

        if function is None:
            return compile_function
        return compile_function(function)

    def compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        if not examples or not callable(function):
            return function

        compiled_any = False
        lower = getattr(function, "lower", None)
        if callable(lower):
            for example in examples:
                arguments = example if isinstance(example, tuple) else (example,)
                try:
                    lower(*arguments).compile()
                except Exception:
                    continue
                else:
                    compiled_any = True
            if self.strict and not compiled_any:
                raise RuntimeError("jax backend could not compile the requested examples.")
            return function

        for example in examples:
            arguments = example if isinstance(example, tuple) else (example,)
            try:
                function(*arguments)
            except Exception:
                continue
            else:
                compiled_any = True
        if self.strict and not compiled_any:
            raise RuntimeError("jax backend could not compile the requested examples.")
        return function


__all__ = ["AcceleratorJax"]
