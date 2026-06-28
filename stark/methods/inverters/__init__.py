"""Built-in linear inverse-action operators.

Inverters solve or approximate the linear correction systems requested by
linearized resolvents. Dense inversion is the small-system default. Krylov and
relaxation families are public for matrix-free and iterative experimentation,
but both still need stronger preconditioning and benchmark evidence before
they become first-contact recommendations.
"""

from stark.methods.inverters.dense import InverterDense
from stark.methods.inverters.krylov import (
    InverterKrylovArnoldi,
    PreconditionerDiagonalInverse,
    PreconditionerNone,
)
from stark.methods.inverters.relaxation import (
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationSpecialist,
    InverterRelaxationStencil,
    InverterRelaxationStencilUpdate,
)

from stark.methods.inverters.configuration import InverterConfiguration
from stark.methods.inverters.support import InverterDescriptor

__all__ = [
    "InverterConfiguration",
    "InverterDescriptor",
    "InverterDense",
    "InverterKrylovArnoldi",
    "PreconditionerDiagonalInverse",
    "PreconditionerNone",
    "InverterRelaxationJacobi",
    "InverterRelaxationRichardson",
    "InverterRelaxationSpecialist",
    "InverterRelaxationStencil",
    "InverterRelaxationStencilUpdate",
]
