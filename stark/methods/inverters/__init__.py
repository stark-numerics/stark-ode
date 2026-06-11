"""Built-in linear inverse-action operators."""

from stark.methods.inverters.relaxation import (
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationSpecialist,
    InverterRelaxationStencil,
    InverterRelaxationStencilUpdate,
)

from stark.methods.inverters.legacy.bicgstab import InverterBiCGStab
from stark.methods.inverters.legacy.fgmres import InverterFGMRES
from stark.methods.inverters.legacy.gmres import InverterGMRES
from stark.methods.inverters.configuration import InverterConfiguration
from stark.methods.inverters.legacy_support.descriptor import InverterDescriptor
from stark.methods.inverters.legacy_support.adapter import InverterLegacyAdapter
from stark.methods.inverters.legacy_support.policy import InverterPolicy

__all__ = [
    "InverterBiCGStab",
    "InverterConfiguration",
    "InverterDescriptor",
    "InverterFGMRES",
    "InverterGMRES",
    "InverterLegacyAdapter",
    "InverterPolicy",
    "InverterRelaxationJacobi",
    "InverterRelaxationRichardson",
    "InverterRelaxationSpecialist",
    "InverterRelaxationStencil",
    "InverterRelaxationStencilUpdate",
]
