from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar, cast

from stark.accelerators.none import AcceleratorNone
from stark.contracts.accelerator import Accelerator

KernelType = TypeVar("KernelType", bound=Callable[..., object])


@dataclass(frozen=True, slots=True)
class AlgebraistGeneratorCompiler:
    """Compile a generated source string into a callable kernel."""

    accelerator: Accelerator = field(default_factory=AcceleratorNone)

    def compile(self, source: str) -> KernelType:
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

            return cast(KernelType, kernel)

        kernel = namespace["kernel"]
        if not callable(kernel):
            raise TypeError("generated kernel is not callable.")

        return cast(
            KernelType,
            self.accelerator.compile(kernel, label="algebraist.generator.kernel"),
        )
