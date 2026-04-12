"""Top-level package for stark-ode."""

from stark.butcher_tableau import ButcherTableau, EmbeddedButcherTableau
from stark.inverter_support.block_operator import BlockOperator
from stark.inverter_support.inversion import Inversion
from stark.inverter_support.descriptor import InverterDescriptor
from stark.inverter_library import InverterGMRES
from stark.marcher import Marcher
from stark.audit import AuditError, Auditor
from stark.control import Regulator, Tolerance
from stark.contracts import (
    Block,
    Derivative,
    InnerProduct,
    InverterLike,
    IntervalLike,
    LinearResidual,
    Linearizer,
    Operator,
    Residual,
    ResolverLike,
    Scheme,
    SchemeLike,
    Translation,
    Workbench,
)
from stark.integrate import Integrator
from stark.resolver_support.resolution import Resolution
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_library import ResolverNewton, ResolverPicard
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.primitives import Interval
from stark.scheme_support.workspace import SchemeWorkspace

__all__ = [
    "Block",
    "BlockOperator",
    "ButcherTableau",
    "Marcher",
    "AuditError",
    "Auditor",
    "Derivative",
    "EmbeddedButcherTableau",
    "InnerProduct",
    "Inversion",
    "Integrator",
    "InverterDescriptor",
    "InverterGMRES",
    "InverterLike",
    "Interval",
    "IntervalLike",
    "LinearResidual",
    "Linearizer",
    "Operator",
    "Residual",
    "Resolution",
    "ResolverDescriptor",
    "ResolverLike",
    "ResolverNewton",
    "ResolverPicard",
    "Scheme",
    "SchemeDescriptor",
    "SchemeLike",
    "SchemeWorkspace",
    "Regulator",
    "Tolerance",
    "Translation",
    "Workbench",
]
