"""Public contract index for user-supplied STARK objects.

The submodules are the navigable documentation layer. This package-level module
re-exports the most common protocol names for examples and interactive use.
Prefer focused submodule imports in implementation code.
"""

from stark.core.contracts.accelerator import Accelerator
from stark.core.contracts.block import (
    BlockLike,
    BlockOperatorDiagonalLike,
    BlockOperatorEntryLike,
    BlockOperatorLike,
)
from stark.core.contracts.carrier import (
    Carrier,
    CarrierAllocation,
    CarrierArithmetic,
    CarrierNorm,
    CarrierStorage,
    CarrierValidation,
)
from stark.core.contracts.derivative import DerivativeAudit, DerivativeLike
from stark.core.contracts.derivative_split import DerivativeSplitAudit, DerivativeSplitLike
from stark.core.contracts.errors import StarkError, StarkErrorRecoverable
from stark.core.contracts.engine import Engine
from stark.core.contracts.inner_product import InnerProduct
from stark.core.contracts.integrator import IntegratorLike
from stark.core.contracts.interval import IntervalLike
from stark.core.contracts.inverter import (
    Inverter,
    InverterInstance,
    InverterInstancing,
    InverterOutputMode,
    InverterRequest,
)
from stark.core.contracts.linear_combine import (
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
from stark.core.contracts.linearizer import LinearizerLike, LinearizerAudit
from stark.core.contracts.stepper import IntegratorStepperAudit, IntegratorStepperLike
from stark.core.contracts.operator import Operator, OperatorType
from stark.core.contracts.residual import LinearResidual, Residual, ResidualAudit
from stark.core.contracts.resolvent import Resolvent, ResolventAudit
from stark.core.contracts.scheme import Scheme, SchemeAudit, SchemeLike
from stark.core.contracts.scheme_predictor import SchemePredictorLike
from stark.core.contracts.state import (
    State,
    StateType,
    StateTypeCovariant,
    StateTypeContravariant,
)
from stark.core.contracts.translation import TranslationAudit
from stark.core.contracts.translation_basis import TranslationBasis
from stark.core.contracts.translation import (
    Translation,
    TranslationType,
    TranslationTypeCovariant,
    TranslationTypeContravariant,
)
from stark.core.contracts.allocator import Allocator, AllocatorAudit

__all__ = [
    "Accelerator",
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
    "DerivativeAudit",
    "DerivativeLike",
    "DerivativeSplitAudit",
    "DerivativeSplitLike",
    "Engine",
    "StarkError",
    "StarkErrorRecoverable",
    "InnerProduct",
    "IntegratorLike",
    "IntervalLike",
    "Inverter",
    "InverterInstance",
    "InverterInstancing",
    "InverterOutputMode",
    "InverterRequest",
    "LinearCombine",
    "LinearResidual",
    "LinearizerLike",
    "LinearizerAudit",
    "IntegratorStepperAudit",
    "IntegratorStepperLike",
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
    "SchemePredictorLike",
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
