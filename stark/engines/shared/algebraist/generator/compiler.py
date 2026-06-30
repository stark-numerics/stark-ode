from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from stark.engines.shared.accelerators.none import AcceleratorNone
from stark.core.contracts.accelerator import Accelerator

_COMPILED_KERNELS: dict[tuple[object, ...], Callable[..., object]] = {}


def _accelerator_key(accelerator: Accelerator) -> tuple[object, ...]:
    return (
        type(accelerator).__module__,
        type(accelerator).__qualname__,
        getattr(accelerator, "name", None),
        getattr(accelerator, "cache", None),
        getattr(accelerator, "strict", None),
        tuple(
            (name, repr(value))
            for name, value in sorted(getattr(accelerator, "options", {}).items())
        ),
    )


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorCompiler:
    """Compile a generated source string into a callable kernel."""

    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def compile(self, source: str) -> Callable[..., object]:
        cache_key = (*_accelerator_key(self.accelerator), source)
        cached = _COMPILED_KERNELS.get(cache_key)
        if cached is not None:
            return cached

        namespace: dict[str, object] = {}
        exec(source, namespace)

        if "kernel" not in namespace:
            raise ValueError("generated source must define a function named 'kernel'.")

        if "_kernel_flat" in namespace:
            flat = namespace["_kernel_flat"]
            if not callable(flat):
                raise TypeError("generated _kernel_flat is not callable.")

            namespace["_kernel_flat"] = self.accelerator.compile(
                flat,
                label="algebraist.generator.flat",
            )

            kernel = namespace["kernel"]
            if not callable(kernel):
                raise TypeError("generated kernel is not callable.")

            _COMPILED_KERNELS[cache_key] = kernel
            return kernel

        kernel = namespace["kernel"]
        if not callable(kernel):
            raise TypeError("generated kernel is not callable.")

        compiled = cast(
            Callable[..., object],
            self.accelerator.compile(kernel, label="algebraist.generator.kernel"),
        )
        _COMPILED_KERNELS[cache_key] = compiled
        return compiled
