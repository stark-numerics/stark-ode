"""Arithmetic policies for prepared STARK carriers."""

from stark.carriers.native.arithmetic import CarrierArithmeticNative
from stark.carriers.numpy.arithmetic import CarrierArithmeticNumpy

__all__ = [
    "CarrierArithmeticNative",
    "CarrierArithmeticNumpy",
]

try:
    from stark.carriers.cupy.arithmetic import CarrierArithmeticCupy
except ImportError:
    pass
else:
    __all__ += ["CarrierArithmeticCupy"]

try:
    from stark.carriers.jax.arithmetic import CarrierArithmeticJax
except ImportError:
    pass
else:
    __all__ += ["CarrierArithmeticJax"]