"""Generated and runtime algebra providers for STARK hot paths.

Algebraist objects provide explicit kernels for vector-space operations used
inside schemes and related workers. General providers supply arity-based
translation combinations; specialist providers supply fixed-coefficient
stencils for scheme stages, accepted increments, and embedded error estimates.

This is an advanced contributor surface. Backend implementations use it to
prepare fast algebra for known `Frame` layouts. Runtime providers remain useful
for unknown or foreign state shapes, but generated providers are the intended
path for ordinary high-level `Frame` models.
"""

from stark.engines.shared.algebraist.algebraist import Algebraist, AlgebraistKernel, AlgebraistRequest
from stark.engines.shared.algebraist.arity import AlgebraistArity
from stark.engines.shared.algebraist.linear_combine import AlgebraistLinearCombine, AlgebraistLinearCombineKernel
from stark.engines.shared.algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.engines.shared.algebraist.frame import (
    MAX_UNRAVEL_SIZE,
    AlgebraistFrame,
    AlgebraistFrameBroadcast,
    AlgebraistField,
    AlgebraistFrameLooped,
    AlgebraistNormExcluded,
    AlgebraistNormLike,
    AlgebraistNormMax,
    AlgebraistNormRMS,
    AlgebraistFieldPath,
    AlgebraistFramePolicy,
    AlgebraistFrameScalar,
    AlgebraistFrameUnravel,
)
from stark.engines.shared.algebraist.norm import AlgebraistNorm, AlgebraistNormKernel
from stark.engines.shared.algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.engines.shared.algebraist.specialist import AlgebraistSpecialist
from stark.engines.shared.algebraist.stencil import AlgebraistStencil
from stark.engines.shared.algebraist.allocator import AlgebraistAllocator

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
    "AlgebraistField",
    "AlgebraistFrameLooped",
    "AlgebraistNormExcluded",
    "AlgebraistNormLike",
    "AlgebraistNormMax",
    "AlgebraistNormRMS",
    "AlgebraistFieldPath",
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
