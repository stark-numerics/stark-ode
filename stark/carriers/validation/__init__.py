"""Validation policies for prepared STARK carriers."""

from stark.carriers.native.validation import CarrierValidationNative
from stark.carriers.numpy.validation import CarrierValidationNumpy

__all__ = [
    "CarrierValidationNative",
    "CarrierValidationNumpy",
]

try:
    from stark.carriers.cupy.validation import CarrierValidationCupy
except ImportError:
    pass
else:
    __all__ += ["CarrierValidationCupy"]

try:
    from stark.carriers.jax.validation import CarrierValidationJax
except ImportError:
    pass
else:
    __all__ += ["CarrierValidationJax"]