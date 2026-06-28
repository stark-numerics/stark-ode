"""Stationary relaxation inverter family.

Relaxation inverters such as Richardson and Jacobi are simple iterative
inverse-action approximations. They are useful for teaching, preconditioner
experiments, and problems where the operator structure makes a cheap iteration
competitive. They are not currently the main performance showcase.
"""

from stark.methods.inverters.relaxation.jacobi import (
    InverterRelaxationJacobi,
    InverterRelaxationJacobiInverse,
)
from stark.methods.inverters.relaxation.richardson import InverterRelaxationRichardson
from stark.methods.inverters.relaxation.specialist import InverterRelaxationSpecialist
from stark.methods.inverters.relaxation.stencil import InverterRelaxationStencil, InverterRelaxationStencilUpdate

__all__ = [
    "InverterRelaxationJacobi",
    "InverterRelaxationJacobiInverse",
    "InverterRelaxationRichardson",
    "InverterRelaxationSpecialist",
    "InverterRelaxationStencil",
    "InverterRelaxationStencilUpdate",
]
