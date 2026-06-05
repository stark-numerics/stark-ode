"""Built-in linear inverse-action operators."""

from stark.inverters.relaxation import (
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationSpecialist,
    InverterRelaxationStencil,
    InverterRelaxationStencilUpdate,
)

from stark.inverters.legacy.bicgstab import InverterBiCGStab
from stark.inverters.legacy.fgmres import InverterFGMRES
from stark.inverters.legacy.gmres import InverterGMRES
from stark.inverters.configuration import InverterConfiguration
from stark.inverters.legacy_support.descriptor import InverterDescriptor
from stark.inverters.legacy_support.policy import InverterPolicy

__all__ = [
    "InverterBiCGStab",
    "InverterConfiguration",
    "InverterDescriptor",
    "InverterFGMRES",
    "InverterGMRES",
    "InverterPolicy",
    "InverterRelaxationJacobi",
    "InverterRelaxationRichardson",
    "InverterRelaxationSpecialist",
    "InverterRelaxationStencil",
    "InverterRelaxationStencilUpdate",
]
