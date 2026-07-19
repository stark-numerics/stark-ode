"""Public contract index for user-supplied STARK objects.

The submodules are the navigable documentation layer. This package-level module
re-exports the most common protocol names for examples and interactive use.
Prefer focused submodule imports in implementation code.
"""

from stark.core.contracts.engines.accelerator import Accelerator
from stark.core.contracts.methods.block import (
    BlockLike,
    BlockOperatorDiagonalLike,
    BlockOperatorEntryLike,
    BlockOperatorLike,
)
from stark.core.contracts.engines.carrier import (
    CarrierLike,
    CarrierAllocationLike,
    CarrierArithmeticLike,
    CarrierNormLike,
    CarrierStorageLike,
    CarrierValidationLike,
)
from stark.core.contracts.problem.dynamics import DynamicsAudit, DynamicsLike
from stark.core.contracts.problem.dynamics_split import DynamicsSplitAudit, DynamicsSplitLike
from stark.core.contracts.shared.errors import StarkError, StarkErrorRecoverable
from stark.core.contracts.engines.engine import EngineLike
from stark.core.contracts.problem.field import (
    FieldLike,
    FieldPath,
    FieldPathLike,
    FieldPolicyLike,
    FieldPolicyType,
)
from stark.core.contracts.problem.frame import FrameLike
from stark.core.contracts.problem.inner_product import InnerProduct, InnerProductNamed
from stark.core.contracts.methods.integrator import IntegratorLike
from stark.core.contracts.shared.interval import IntervalLike
from stark.core.contracts.problem.norm import NormLike
from stark.core.contracts.methods.inverter import (
    Inverter,
    InverterInstance,
    InverterInstancing,
    InverterOutputMode,
    InverterRequest,
)
from stark.core.contracts.engines.linear_combine import (
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
from stark.core.contracts.problem.linearizer import LinearizerLike, LinearizerAudit
from stark.core.contracts.methods.stepper import IntegratorStepperAudit, IntegratorStepperLike
from stark.core.contracts.methods.translation_operator import TranslationOperator, TranslationOperatorType
from stark.core.contracts.methods.residual import LinearResidual, Residual, ResidualAudit
from stark.core.contracts.methods.resolvent import Resolvent, ResolventAudit
from stark.core.contracts.methods.scheme import Scheme, SchemeAudit, SchemeLike
from stark.core.contracts.methods.scheme_predictor import SchemePredictorLike
from stark.core.contracts.problem.state import (
    State,
    StateType,
    StateTypeCovariant,
    StateTypeContravariant,
)
from stark.core.contracts.problem.translation import TranslationAudit
from stark.core.contracts.engines.translation_basis import TranslationBasisLike
from stark.core.contracts.problem.translation import (
    Translation,
    TranslationFieldType,
    TranslationFieldTypeCovariant,
    TranslationFieldTypeContravariant,
    TranslationType,
    TranslationTypeCovariant,
    TranslationTypeContravariant,
)
from stark.core.contracts.engines.allocator import AllocatorLike, AllocatorAudit

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
    "EngineLike",
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
    "TranslationOperator",
    "TranslationOperatorType",
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
    "AllocatorLike",
    "AllocatorAudit",
]
