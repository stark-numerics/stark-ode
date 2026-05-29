"""Public contract index for user-supplied STARK objects.

The submodules are the navigable documentation layer. This package-level module
re-exports the most common protocol names for examples and interactive use.
Prefer focused submodule imports in implementation code.
"""

from stark.contracts.acceleration import AccelerationBackend, AcceleratorLike
from stark.contracts.blocks import Block
from stark.contracts.carriers import (
    Carrier,
    CarrierAllocation,
    CarrierArithmetic,
    CarrierNorm,
    CarrierStorage,
    CarrierValidation,
)
from stark.contracts.derivative_imex import DerivativeIMEX, DerivativeIMEXAudit
from stark.contracts.derivatives import Derivative, DerivativeAudit
from stark.contracts.errors import StarkError, StarkErrorRecoverable
from stark.contracts.inner_products import InnerProduct
from stark.contracts.integrators import IntegratorLike
from stark.contracts.intervals import IntervalLike
from stark.contracts.inverters import (
    InverterAudit,
    InverterLike,
    InverterPreconditionerLike,
)
from stark.contracts.linear_combine import (
    Combine2,
    Combine3,
    Combine4,
    Combine5,
    Combine6,
    Combine7,
    Combine8,
    Combine9,
    Combine10,
    Combine11,
    Combine12,
    LinearCombine,
    Scale,
    SupportsLinearCombine,
)
from stark.contracts.linearizers import Linearizer, LinearizerAudit
from stark.contracts.marchers import MarcherAudit, MarcherLike
from stark.contracts.operators import Operator
from stark.contracts.residuals import LinearResidual, Residual, ResidualAudit
from stark.contracts.resolvents import Resolvent, ResolventAudit
from stark.contracts.schemes import Scheme, SchemeAudit, SchemeLike
from stark.contracts.states import (
    State,
    StateType,
    StateTypeCovariant,
    StateTypeContravariant,
)
from stark.contracts.translation_audit import TranslationAudit
from stark.contracts.translations import (
    Translation,
    TranslationType,
    TranslationTypeCovariant,
    TranslationTypeContravariant,
)
from stark.contracts.allocators import Allocator, AllocatorAudit

__all__ = [
    "AccelerationBackend",
    "AcceleratorLike",
    "Block",
    "Carrier",
    "CarrierAllocation",
    "CarrierArithmetic",
    "CarrierNorm",
    "CarrierStorage",
    "CarrierValidation",
    "Combine2",
    "Combine3",
    "Combine4",
    "Combine5",
    "Combine6",
    "Combine7",
    "Combine8",
    "Combine9",
    "Combine10",
    "Combine11",
    "Combine12",
    "Derivative",
    "DerivativeIMEX",
    "DerivativeIMEXAudit",
    "DerivativeAudit",
    "StarkError",
    "StarkErrorRecoverable",
    "InnerProduct",
    "IntegratorLike",
    "IntervalLike",
    "InverterAudit",
    "InverterLike",
    "InverterPreconditionerLike",
    "LinearCombine",
    "LinearResidual",
    "Linearizer",
    "LinearizerAudit",
    "MarcherAudit",
    "MarcherLike",
    "Operator",
    "Residual",
    "ResidualAudit",
    "Resolvent",
    "ResolventAudit",
    "Scale",
    "Scheme",
    "SchemeAudit",
    "SchemeLike",
    "State",
    "StateType",
    "StateTypeCovariant",
    "StateTypeContravariant",
    "SupportsLinearCombine",
    "Translation",
    "TranslationAudit",
    "TranslationType",
    "TranslationTypeCovariant",
    "TranslationTypeContravariant",
    "Allocator",
    "AllocatorAudit",
]
