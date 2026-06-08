"""Generated and runtime algebra providers for STARK hot paths.

Algebraist objects provide explicit kernels for vector-space operations used
inside schemes and related workers. General providers supply arity-based
translation combinations; specialist providers supply fixed-coefficient
stencils for scheme stages, accepted increments, and embedded error estimates.
"""

from stark.algebraist.algebraist import Algebraist, AlgebraistKernel, AlgebraistRequest
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.linear_combine import AlgebraistLinearCombine, AlgebraistLinearCombineKernel
from stark.algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
)
from stark.algebraist.layout import (
    MAX_UNRAVEL_SIZE,
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
    AlgebraistLayoutNormExcluded,
    AlgebraistLayoutNormMax,
    AlgebraistLayoutNormPolicy,
    AlgebraistLayoutNormRMS,
    AlgebraistLayoutPath,
    AlgebraistLayoutPolicy,
    AlgebraistLayoutScalar,
    AlgebraistLayoutUnravel,
)
from stark.algebraist.norm import AlgebraistNorm, AlgebraistNormKernel
from stark.algebraist.runtime import (
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeSpecialist,
)
from stark.algebraist.specialist import AlgebraistSpecialist
from stark.algebraist.stencil import AlgebraistStencil
from stark.algebraist.allocator import AlgebraistAllocator

__all__ = [
    "MAX_UNRAVEL_SIZE",
    "Algebraist",
    "AlgebraistArity",
    "AlgebraistLinearCombine",
    "AlgebraistLinearCombineKernel",
    "AlgebraistGeneratorCompiler",
    "AlgebraistGeneratorEmitter",
    "AlgebraistGeneratorLinearCombine",
    "AlgebraistGeneratorNorm",
    "AlgebraistGeneratorSpecialist",
    "AlgebraistKernel",
    "AlgebraistLayout",
    "AlgebraistLayoutBroadcast",
    "AlgebraistLayoutField",
    "AlgebraistLayoutLooped",
    "AlgebraistLayoutNormExcluded",
    "AlgebraistLayoutNormMax",
    "AlgebraistLayoutNormPolicy",
    "AlgebraistLayoutNormRMS",
    "AlgebraistLayoutPath",
    "AlgebraistLayoutPolicy",
    "AlgebraistLayoutScalar",
    "AlgebraistLayoutUnravel",
    "AlgebraistNorm",
    "AlgebraistNormKernel",
    "AlgebraistRequest",
    "AlgebraistRuntimeLinearCombine",
    "AlgebraistRuntimeNorm",
    "AlgebraistRuntimeSpecialist",
    "AlgebraistSpecialist",
    "AlgebraistStencil",
    "AlgebraistAllocator",
]
