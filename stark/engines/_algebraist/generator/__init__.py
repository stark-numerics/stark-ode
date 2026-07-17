"""Generated Algebraist providers for known `Frame` layouts.

Generator providers emit backend-shaped kernels for algebra that schemes,
resolvents, and inverters call repeatedly. Backend authors usually customize
the target rather than bypassing this package: NumPy, CuPy, JAX, and future
Torch engines may want different expression styles while preserving the same
high-level `Frame` contract.
"""

from stark.engines._algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines._algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines._algebraist.generator.inner_product import AlgebraistGeneratorInnerProduct
from stark.engines._algebraist.generator.linear_combine import AlgebraistGeneratorLinearCombine
from stark.engines._algebraist.generator.norm import AlgebraistGeneratorNorm
from stark.engines._algebraist.generator.linear_fixed import AlgebraistGeneratorLinearFixed
from stark.engines._algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetFunctional,
    AlgebraistGeneratorTargetMutable,
    AlgebraistGeneratorTargetMutableVectorized,
)
from stark.engines._algebraist.generator.target_cupy import AlgebraistGeneratorTargetCupy

__all__ = [
    "AlgebraistGeneratorCompiler",
    "AlgebraistGeneratorEmitter",
    "AlgebraistGeneratorInnerProduct",
    "AlgebraistGeneratorLinearCombine",
    "AlgebraistGeneratorNorm",
    "AlgebraistGeneratorLinearFixed",
    "AlgebraistGeneratorTarget",
    "AlgebraistGeneratorTargetFunctional",
    "AlgebraistGeneratorTargetMutable",
    "AlgebraistGeneratorTargetMutableVectorized",
    "AlgebraistGeneratorTargetCupy",
]
