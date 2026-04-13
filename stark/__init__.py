"""Top-level package for stark-ode."""

from stark.butcher_tableau import ButcherTableau, EmbeddedButcherTableau
from stark.inverter_support.block_operator import BlockOperator
from stark.inverter_support.descriptor import InverterDescriptor
from stark.inverter_support.policy import InverterPolicy
from stark.inverter_support.tolerance import InverterTolerance
from stark.inverter_library import InverterBiCGStab, InverterFGMRES, InverterGMRES
from stark.marcher import Marcher
from stark.audit import AuditError, Auditor
from stark.regulator import Regulator
from stark.tolerance import Tolerance
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
from stark.resolver_support.descriptor import ResolverDescriptor
from stark.resolver_support.policy import ResolverPolicy
from stark.resolver_support.tolerance import ResolverTolerance
from stark.resolver_library import ResolverAnderson, ResolverBroyden, ResolverNewton, ResolverPicard
from stark.safety import Safety
from stark.scheme_support.descriptor import SchemeDescriptor
from stark.scheme_support.tolerance import SchemeTolerance
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
    "InverterPolicy",
    "InverterBiCGStab",
    "Integrator",
    "InverterDescriptor",
    "InverterFGMRES",
    "InverterGMRES",
    "InverterLike",
    "InverterTolerance",
    "Interval",
    "IntervalLike",
    "LinearResidual",
    "Linearizer",
    "Operator",
    "Residual",
    "ResolverAnderson",
    "ResolverBroyden",
    "ResolverDescriptor",
    "ResolverLike",
    "ResolverNewton",
    "ResolverPicard",
    "ResolverPolicy",
    "ResolverTolerance",
    "Safety",
    "Scheme",
    "SchemeDescriptor",
    "SchemeLike",
    "SchemeTolerance",
    "SchemeWorkspace",
    "Regulator",
    "Tolerance",
    "Translation",
    "Workbench",
]

