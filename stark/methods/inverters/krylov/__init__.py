"""Matrix-free Krylov inverter family.

Krylov inverters approximate inverse actions using operator applications
rather than materialising the full operator. They are intended for larger or
operator-only linear systems, but this package is still in development:
preconditioner coverage is deliberately small and contributors with real
matrix-free workloads can help shape the public defaults.
"""

from stark.methods.inverters.krylov.arnoldi import (
    InverterKrylovArnoldi,
    InverterKrylovArnoldiInstance,
)
from stark.methods.inverters.krylov.basis import InverterKrylovBasis
from stark.methods.inverters.krylov.preconditioners import (
    InverterPreconditionerDiagonalInverse,
    InverterPreconditionerNone,
    InverterKrylovPreconditionerLike,
)
from stark.methods.inverters.krylov.projection import InverterKrylovProjection

__all__ = [
    "InverterKrylovArnoldi",
    "InverterKrylovArnoldiInstance",
    "InverterKrylovBasis",
    "InverterKrylovProjection",
    "InverterKrylovPreconditionerLike",
    "InverterPreconditionerDiagonalInverse",
    "InverterPreconditionerNone",
]
