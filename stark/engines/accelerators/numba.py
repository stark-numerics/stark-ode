from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, overload

from stark.core.contracts.accelerator import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorNumba:
    """Numba-backed accelerator for imperative numerical kernels."""

    strict: bool = False
    cache: bool = True
    options: dict[str, Any] = field(default_factory=dict, repr=False)
    _njit: Any = field(init=False, repr=False, default=None)
    _typeof: Any = field(init=False, repr=False, default=None)

    name: ClassVar[str] = "numba"

    def __init__(
        self,
        *,
        cache: bool = True,
        strict: bool = False,
        options: dict[str, Any] | None = None,
    ) -> None:
        try:
            from numba import njit, typeof
        except ImportError:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "AcceleratorNumba requires Numba to be installed."
            ) from None

        self.cache = cache
        self.strict = strict
        self.options = {} if options is None else dict(options)
        self._njit = njit
        self._typeof = typeof

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
        del label
        base_options = {"cache": self.cache, **self.options, **options}
        if cache is not None:
            base_options["cache"] = cache

        def compile_function(target: AcceleratorTarget) -> AcceleratorTarget:
            target_options = dict(base_options)

            code = getattr(target, "__code__", None)
            if code is None:
                if self.strict:
                    raise RuntimeError("numba backend can only compile plain Python functions.")
                return target

            filename = None if code is None else code.co_filename

            # Functions generated with exec(source, namespace) usually report
            # their filename as "<string>". Numba can JIT them, but cannot use
            # its persistent disk cache because there is no file locator.
            if filename == "<string>":
                target_options["cache"] = False

            return self._njit(**target_options)(target)

        if function is None:
            return compile_function

        return compile_function(function)

    def compile_examples(
        self,
        function: AcceleratorTarget,
        *examples: Any,
    ) -> AcceleratorTarget:
        compile_method = getattr(function, "compile", None)
        if not examples or not callable(function) or not callable(compile_method):
            return function

        compiled_any = False
        for example in examples:
            arguments = example if isinstance(example, tuple) else (example,)
            try:
                compile_method(tuple(self._typeof(argument) for argument in arguments))
            except Exception:
                continue
            else:
                compiled_any = True

        if self.strict and not compiled_any:
            raise RuntimeError("numba backend could not compile the requested examples.")

        return function


__all__ = ["AcceleratorNumba"]
