"""Generated and runtime algebra providers for STARK hot paths.

Algebraist objects provide explicit kernels for vector-space operations used
inside schemes and related workers. General providers supply arity-based
translation combinations; linear-fixed providers supply fixed-coefficient
stencils for scheme stages, accepted increments, and embedded error estimates.

This is an advanced contributor surface. Backend implementations use it to
prepare fast algebra for known `Frame` layouts. Runtime providers remain useful
for unknown or foreign state shapes, but generated providers are the intended
path for ordinary high-level `Frame` models.
"""

from stark.engines._algebraist.algebraist import Algebraist
from stark.engines._algebraist.arity import AlgebraistArity
from stark.engines._algebraist.inner_product import (
    AlgebraistInnerProduct,
    AlgebraistInnerProductKernel,
)
from stark.engines._algebraist.linear_combine import (
    AlgebraistLinearCombine,
    AlgebraistLinearCombineKernel,
)
from stark.engines._algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorInnerProduct,
    AlgebraistGeneratorLinearCombine,
    AlgebraistGeneratorNorm,
    AlgebraistGeneratorLinearFixed,
)
from stark.engines._algebraist.norm import AlgebraistNorm, AlgebraistNormKernel
from stark.engines._algebraist.runtime import (
    AlgebraistRuntimeInnerProduct,
    AlgebraistRuntimeLinearCombine,
    AlgebraistRuntimeNorm,
    AlgebraistRuntimeLinearFixed,
)
from stark.engines._algebraist.linear_fixed import AlgebraistLinearFixed
from stark.engines._algebraist.stencil import AlgebraistStencil
from stark.engines._algebraist.allocator import AlgebraistAllocator

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
    "AlgebraistGeneratorLinearFixed",
    "AlgebraistInnerProduct",
    "AlgebraistInnerProductKernel",
    "AlgebraistNorm",
    "AlgebraistNormKernel",
    "AlgebraistRuntimeInnerProduct",
    "AlgebraistRuntimeLinearCombine",
    "AlgebraistRuntimeNorm",
    "AlgebraistRuntimeLinearFixed",
    "AlgebraistLinearFixed",
    "AlgebraistStencil",
    "AlgebraistAllocator",
]
