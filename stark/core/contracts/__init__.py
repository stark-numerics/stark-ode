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
    CarrierLike,
    CarrierAllocationLike,
    CarrierArithmeticLike,
    CarrierNormLike,
    CarrierStorageLike,
    CarrierValidationLike,
)
from stark.core.contracts.dynamics import DynamicsAudit, DynamicsLike
from stark.core.contracts.dynamics_split import DynamicsSplitAudit, DynamicsSplitLike
from stark.core.contracts.errors import StarkError, StarkErrorRecoverable
from stark.core.contracts.engine import Engine
from stark.core.contracts.field import (
    FieldLike,
    FieldPath,
    FieldPathLike,
    FieldPolicyLike,
    FieldPolicyType,
)
from stark.core.contracts.frame import FrameLike
from stark.core.contracts.inner_product import InnerProduct, InnerProductNamed
from stark.core.contracts.integrator import IntegratorLike
from stark.core.contracts.interval import IntervalLike
from stark.core.contracts.norm import NormLike
from stark.core.contracts.inverter import (
    Inverter,
    InverterInstance,
    InverterInstancing,
    InverterOutputMode,
    InverterRequest,
)
from stark.core.contracts.linear_combine import (
    LinearCombine,
    LinearCombineArity2Like,
    LinearCombineArity3Like,
    LinearCombineArity4Like,
    LinearCombineArity5Like,
    LinearCombineArity6Like,
    LinearCombineArity7Like,
    LinearCombineArity8Like,
    LinearCombineArity9Like,
    LinearCombineArity10Like,
    LinearCombineArity11Like,
    LinearCombineArity12Like,
    LinearCombineScaleLike,
    LinearCombineSupporting,
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
from stark.core.contracts.translation_basis import TranslationBasisLike
from stark.core.contracts.translation import (
    Translation,
    TranslationFieldType,
    TranslationFieldTypeCovariant,
    TranslationFieldTypeContravariant,
    TranslationType,
    TranslationTypeCovariant,
    TranslationTypeContravariant,
)
from stark.core.contracts.translation_factory import TranslationFactoryLike
from stark.core.contracts.allocator import AllocatorLike, AllocatorAudit

__all__ = [
    "Accelerator",
    "BlockLike",
    "BlockOperatorEntryLike",
    "BlockOperatorLike",
    "BlockOperatorDiagonalLike",
    "CarrierLike",
    "CarrierAllocationLike",
    "CarrierArithmeticLike",
    "CarrierNormLike",
    "CarrierStorageLike",
    "CarrierValidationLike",
    "DynamicsAudit",
    "DynamicsLike",
    "DynamicsSplitAudit",
    "DynamicsSplitLike",
    "Engine",
    "FieldLike",
    "FieldPath",
    "FieldPathLike",
    "FieldPolicyLike",
    "FieldPolicyType",
    "FrameLike",
    "StarkError",
    "StarkErrorRecoverable",
    "InnerProduct",
    "InnerProductNamed",
    "IntegratorLike",
    "IntervalLike",
    "Inverter",
    "InverterInstance",
    "InverterInstancing",
    "InverterOutputMode",
    "InverterRequest",
    "LinearCombine",
    "LinearCombineArity2Like",
    "LinearCombineArity3Like",
    "LinearCombineArity4Like",
    "LinearCombineArity5Like",
    "LinearCombineArity6Like",
    "LinearCombineArity7Like",
    "LinearCombineArity8Like",
    "LinearCombineArity9Like",
    "LinearCombineArity10Like",
    "LinearCombineArity11Like",
    "LinearCombineArity12Like",
    "LinearCombineScaleLike",
    "LinearCombineSupporting",
    "LinearResidual",
    "LinearizerLike",
    "LinearizerAudit",
    "NormLike",
    "IntegratorStepperAudit",
    "IntegratorStepperLike",
    "Operator",
    "OperatorType",
    "Residual",
    "ResidualAudit",
    "Resolvent",
    "ResolventAudit",
    "Scheme",
    "SchemeAudit",
    "SchemeLike",
    "SchemePredictorLike",
    "State",
    "StateType",
    "StateTypeCovariant",
    "StateTypeContravariant",
    "Translation",
    "TranslationAudit",
    "TranslationBasisLike",
    "TranslationFieldType",
    "TranslationFieldTypeCovariant",
    "TranslationFieldTypeContravariant",
    "TranslationType",
    "TranslationTypeCovariant",
    "TranslationTypeContravariant",
    "TranslationFactoryLike",
    "AllocatorLike",
    "AllocatorAudit",
]
