"""Generated Algebraist providers for known `Frame` layouts.

Generator providers emit backend-shaped kernels for algebra that schemes,
resolvents, and inverters call repeatedly. Backend authors usually customize
the target rather than bypassing this package: NumPy, CuPy, JAX, and future
Torch engines may want different expression styles while preserving the same
high-level `Frame` contract.
"""

from stark.engines.shared.algebraist.generator.compiler import AlgebraistGeneratorCompiler
from stark.engines.shared.algebraist.generator.emitter import AlgebraistGeneratorEmitter
from stark.engines.shared.algebraist.generator.inner_product import AlgebraistGeneratorInnerProduct
from stark.engines.shared.algebraist.generator.linear_combine import AlgebraistGeneratorLinearCombine
from stark.engines.shared.algebraist.generator.norm import AlgebraistGeneratorNorm
from stark.engines.shared.algebraist.generator.specialist import AlgebraistGeneratorSpecialist
from stark.engines.shared.algebraist.generator.target import (
    AlgebraistGeneratorTarget,
    AlgebraistGeneratorTargetFunctional,
    AlgebraistGeneratorTargetMutable,
    AlgebraistGeneratorTargetMutableVectorized,
)

__all__ = [
    "AlgebraistGeneratorCompiler",
    "AlgebraistGeneratorEmitter",
    "AlgebraistGeneratorInnerProduct",
    "AlgebraistGeneratorLinearCombine",
    "AlgebraistGeneratorNorm",
    "AlgebraistGeneratorSpecialist",
    "AlgebraistGeneratorTarget",
    "AlgebraistGeneratorTargetFunctional",
    "AlgebraistGeneratorTargetMutable",
    "AlgebraistGeneratorTargetMutableVectorized",
]
