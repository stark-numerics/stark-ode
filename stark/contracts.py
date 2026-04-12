from __future__ import annotations

"""
Public protocol re-exports for the STARK API surface.

The concrete protocol definitions live under `stark.contract_protocols.*` so they can be
maintained in smaller, more focused modules. `stark.contracts` remains as the
single public import location for users and existing internal code.
"""

from stark.contract_protocols import (
    Block,
    Combine2,
    Combine3,
    Combine4,
    Combine5,
    Combine6,
    Combine7,
    Derivative,
    InnerProduct,
    InverterLike,
    IntervalLike,
    LinearResidual,
    Linearizer,
    Operator,
    Residual,
    ResolverLike,
    Scale,
    Scheme,
    SchemeLike,
    State,
    Translation,
    Workbench,
)
from stark.scheme_support.descriptor import SchemeDescriptor

__all__ = [
    "Block",
    "Combine2",
    "Combine3",
    "Combine4",
    "Combine5",
    "Combine6",
    "Combine7",
    "Derivative",
    "InnerProduct",
    "InverterLike",
    "IntervalLike",
    "LinearResidual",
    "Linearizer",
    "Operator",
    "Residual",
    "ResolverLike",
    "Scale",
    "Scheme",
    "SchemeDescriptor",
    "SchemeLike",
    "State",
    "Translation",
    "Workbench",
]
