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

from stark.engines.algebraist.algebraist import Algebraist
from stark.engines.algebraist.arity import AlgebraistArity
from stark.engines.algebraist.inner_product import (
    AlgebraistInnerProduct,
    AlgebraistInnerProductKernel,
)
from stark.engines.algebraist.linear_combine import (
    AlgebraistLinearCombine,
    AlgebraistLinearCombineKernel,
)
from stark.engines.algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorSpecialist,
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
    "AlgebraistInnerProduct",
    "AlgebraistInnerProductKernel",
    "AlgebraistNorm",
    "AlgebraistNormKernel",
    "AlgebraistRuntimeInnerProduct",
    "AlgebraistRuntimeLinearCombine",
    "AlgebraistRuntimeNorm",
    "AlgebraistRuntimeSpecialist",
    "AlgebraistSpecialist",
    "AlgebraistStencil",
    "AlgebraistAllocator",
]
