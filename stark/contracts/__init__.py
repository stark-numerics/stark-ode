"""Public contract index for user-supplied STARK objects.

The submodules are the navigable documentation layer. This package-level module
re-exports the most common protocol names for examples and interactive use.
Prefer focused submodule imports in implementation code.
"""

from stark.contracts.accelerator import AcceleratorLike
from stark.contracts.block import (
    BlockLike,
    BlockOperatorDiagonalLike,
    BlockOperatorEntryLike,
    BlockOperatorLike,
)
from stark.contracts.carrier import (
    Carrier,
    CarrierAllocation,
    CarrierArithmetic,
    CarrierNorm,
    CarrierStorage,
    CarrierValidation,
)
from stark.contracts.derivative_imex import DerivativeIMEX, DerivativeIMEXAudit
from stark.contracts.derivative import Derivative, DerivativeAudit
from stark.contracts.errors import StarkError, StarkErrorRecoverable
from stark.contracts.inner_product import InnerProduct
from stark.contracts.integrator import IntegratorLike
from stark.contracts.interval import IntervalLike
from stark.contracts.inverter import (
    Inverter,
    InverterOutputMode,
    InverterRequest,
    LegacyInverterAudit,
    LegacyInverterLike,
    LegacyInverterPreconditionerLike,
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
from stark.contracts.linearizer import Linearizer, LinearizerAudit
from stark.contracts.marcher import MarcherAudit, MarcherLike
from stark.contracts.operator import Operator, OperatorType
from stark.contracts.residual import LinearResidual, Residual, ResidualAudit
from stark.contracts.resolvent import Resolvent, ResolventAudit
from stark.contracts.scheme import Scheme, SchemeAudit, SchemeLike
from stark.contracts.state import (
    State,
    StateType,
    StateTypeCovariant,
    StateTypeContravariant,
)
from stark.contracts.translation import TranslationAudit
from stark.contracts.translation_basis import TranslationBasis
from stark.contracts.translation import (
    Translation,
    TranslationType,
    TranslationTypeCovariant,
    TranslationTypeContravariant,
)
from stark.contracts.allocator import Allocator, AllocatorAudit

__all__ = [
    "AcceleratorLike",
    "BlockLike",
    "BlockOperatorEntryLike",
    "BlockOperatorLike",
    "BlockOperatorDiagonalLike",
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
    "Inverter",
    "InverterOutputMode",
    "InverterRequest",
    "LegacyInverterAudit",
    "LegacyInverterLike",
    "LegacyInverterPreconditionerLike",
    "LinearCombine",
    "LinearResidual",
    "Linearizer",
    "LinearizerAudit",
    "MarcherAudit",
    "MarcherLike",
    "Operator",
    "OperatorType",
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
    "TranslationBasis",
    "TranslationType",
    "TranslationTypeCovariant",
    "TranslationTypeContravariant",
    "Allocator",
    "AllocatorAudit",
]
