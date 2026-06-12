"""Generated and runtime algebra providers for STARK hot paths.

Algebraist objects provide explicit kernels for vector-space operations used
inside schemes and related workers. General providers supply arity-based
translation combinations; specialist providers supply fixed-coefficient
stencils for scheme stages, accepted increments, and embedded error estimates.
"""

from stark.engines.algebraist.algebraist import Algebraist, AlgebraistKernel, AlgebraistRequest
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.linear_combine import AlgebraistLinearCombine, AlgebraistLinearCombineKernel
from stark.engines.algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.algebraist.frame import (
    MAX_UNRAVEL_SIZE,
    AlgebraistFrame,
    AlgebraistFrameBroadcast,
    AlgebraistFrameField,
    AlgebraistFrameLooped,
    AlgebraistFrameNormExcluded,
    AlgebraistFrameNormMax,
    AlgebraistFrameNormPolicy,
    AlgebraistFrameNormRMS,
    AlgebraistFramePath,
    AlgebraistFramePolicy,
    AlgebraistFrameScalar,
    AlgebraistFrameUnravel,
)
from stark.engines.algebraist.norm import AlgebraistNorm, AlgebraistNormKernel
from stark.engines.algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.engines.algebraist.specialist import AlgebraistSpecialist
from stark.engines.algebraist.stencil import AlgebraistStencil
from stark.engines.algebraist.allocator import AlgebraistAllocator

__all__ = [
    "MAX_UNRAVEL_SIZE",
    "Algebraist",
    "AlgebraistArity",
    "AlgebraistLinearCombine",
    "AlgebraistLinearCombineKernel",
    "AlgebraistGeneratorCompiler",
    "AlgebraistGeneratorEmitter",
    "AlgebraistGeneratorInnerProduct",
    "AlgebraistGeneratorLinearCombine",
    "AlgebraistGeneratorNorm",
    "AlgebraistGeneratorSpecialist",
    "AlgebraistKernel",
    "AlgebraistFrame",
    "AlgebraistFrameBroadcast",
    "AlgebraistFrameField",
    "AlgebraistFrameLooped",
    "AlgebraistFrameNormExcluded",
    "AlgebraistFrameNormMax",
    "AlgebraistFrameNormPolicy",
    "AlgebraistFrameNormRMS",
    "AlgebraistFramePath",
    "AlgebraistFramePolicy",
    "AlgebraistFrameScalar",
    "AlgebraistFrameUnravel",
    "AlgebraistNorm",
    "AlgebraistNormKernel",
    "AlgebraistRequest",
    "AlgebraistRuntimeInnerProduct",
    "AlgebraistRuntimeLinearCombine",
    "AlgebraistRuntimeNorm",
    "AlgebraistRuntimeSpecialist",
    "AlgebraistSpecialist",
    "AlgebraistStencil",
    "AlgebraistAllocator",
]
