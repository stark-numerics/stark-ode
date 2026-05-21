"""Allocation policies for prepared STARK carriers."""

from stark.carriers.native.allocation import CarrierAllocationNative
from stark.carriers.numpy.allocation import CarrierAllocationNumpy

__all__ = [
    "CarrierAllocationNative",
    "CarrierAllocationNumpy",
]

try:
    from stark.carriers.cupy.allocation import CarrierAllocationCupy
except ImportError:
    pass
else:
    __all__ += ["CarrierAllocationCupy"]

try:
    from stark.carriers.jax.allocation import CarrierAllocationJax
except ImportError:
    pass
else:
    __all__ += ["CarrierAllocationJax"]