"""Built-in acceleration workers for STARK."""

from stark.accelerators.absent import AcceleratorAbsent
from stark.accelerators.jax import AcceleratorJax
from stark.accelerators.numba import AcceleratorNumba


class Accelerator:
    """Convenience namespace for constructing built-in accelerators."""

    @staticmethod
    def none(*, strict: bool = False) -> AcceleratorAbsent:
        return AcceleratorAbsent(strict=strict)

    @staticmethod
    def absent(*, strict: bool = False) -> AcceleratorAbsent:
        return AcceleratorAbsent(strict=strict)

    @staticmethod
    def numba(*, cache: bool = True, strict: bool = False) -> AcceleratorNumba:
        return AcceleratorNumba(cache=cache, strict=strict)

    @staticmethod
    def jax(*, strict: bool = False) -> AcceleratorJax:
        return AcceleratorJax(strict=strict)


DEFAULT_ACCELERATOR = AcceleratorAbsent()


__all__ = [
    "Accelerator",
    "AcceleratorAbsent",
    "AcceleratorJax",
    "AcceleratorNumba",
    "DEFAULT_ACCELERATOR",
]
