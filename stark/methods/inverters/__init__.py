"""Built-in linear inverse-action operators."""

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
