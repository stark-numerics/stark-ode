from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from stark.core.contracts.engines.accelerator import Accelerator
from stark.engines.accelerators.none import AcceleratorNone

COMPILED_KERNELS: dict[tuple[Any, ...], Callable[..., object]] = {}


def accelerator_key(accelerator: Accelerator) -> tuple[Any, ...]:
    return (
        type(accelerator).__module__,
        type(accelerator).__qualname__,
        accelerator.name,
        getattr(accelerator, "cache", None),
        accelerator.strict,
        tuple(
            (name, repr(value))
            for name, value in sorted(getattr(accelerator, "options", {}).items())
        ),
    )


@dataclass(frozen=True, slots=True)
class GeneratorCompiler:
    """Compile generated source with the selected accelerator."""

    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def compile(self, source: str) -> Callable[..., object]:
        cache_key = (*accelerator_key(self.accelerator), source)
        cached = COMPILED_KERNELS.get(cache_key)
        if cached is not None:
            return cached

        namespace: dict[str, object] = {}
        exec(source, namespace)

        kernel = namespace.get("kernel")
        if not callable(kernel):
            raise TypeError("generated source must define a callable kernel.")

        flat = namespace.get("_kernel_flat")
        if flat is not None:
            if not callable(flat):
                raise TypeError("generated _kernel_flat is not callable.")
            namespace["_kernel_flat"] = self.accelerator.compile(
                flat,
                label="generator.flat",
            )
            COMPILED_KERNELS[cache_key] = kernel
            return kernel

        compiled = cast(
            Callable[..., object],
            self.accelerator.compile(kernel, label="generator.kernel"),
        )
        COMPILED_KERNELS[cache_key] = compiled
        return compiled


__all__ = ["COMPILED_KERNELS", "GeneratorCompiler", "accelerator_key"]
