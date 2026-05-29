from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from stark.accelerators.common import AcceleratorBase
from stark.contracts.acceleration import AcceleratorTarget


@dataclass(slots=True)
class AcceleratorNumba(AcceleratorBase):
    """Numba-backed accelerator for imperative numerical kernels."""

    cache: bool = True
    _njit: Any = field(init=False, repr=False, default=None)
    _typeof: Any = field(init=False, repr=False, default=None)

    name = "numba"

    def __init__(
        self,
        *,
        cache: bool = True,
        strict: bool = False,
        values: dict[str, Any] | None = None,
    ) -> None:
        try:
            from numba import njit, typeof
        except ImportError:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "AcceleratorNumba requires Numba to be installed."
            ) from None

        self.cache = cache
        self.strict = strict
        self.values = {} if values is None else dict(values)
        self._njit = njit
        self._typeof = typeof

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        del label
        base_options = {"cache": self.cache, **self.values, **options}
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

    def _compile_examples(
        self,
        function: AcceleratorTarget,
        *signatures: Any,
    ) -> AcceleratorTarget:
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
