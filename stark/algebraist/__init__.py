"""Generated and runtime algebra providers for STARK hot paths.

Algebraist objects provide explicit kernels for vector-space operations used
inside schemes and related workers. General providers supply arity-based
translation combinations; specialist providers supply fixed-coefficient
stencils for scheme stages, accepted increments, and embedded error estimates.
"""

from stark.algebraist.algebraist import Algebraist, AlgebraistKernel, AlgebraistRequest
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.general import AlgebraistGeneral, AlgebraistGeneralKernel
from stark.algebraist.generator import (
    AlgebraistGeneratorCompiler,
    AlgebraistGeneratorEmitter,
    AlgebraistGeneratorGeneral,
    AlgebraistGeneratorSpecialist,
)
from stark.algebraist.layout import (
    MAX_UNRAVEL_SIZE,
    AlgebraistLayout,
    AlgebraistLayoutBroadcast,
    AlgebraistLayoutField,
    AlgebraistLayoutLooped,
    AlgebraistLayoutPath,
    AlgebraistLayoutPolicy,
    AlgebraistLayoutScalar,
    AlgebraistLayoutUnravel,
)
from stark.algebraist.runtime import AlgebraistRuntimeGeneral, AlgebraistRuntimeSpecialist
from stark.algebraist.specialist import AlgebraistSpecialist
from stark.algebraist.stencil import AlgebraistStencil
from stark.algebraist.allocator import AlgebraistAllocator

__all__ = [
    "MAX_UNRAVEL_SIZE",
    "Algebraist",
    "AlgebraistArity",
    "AlgebraistGeneral",
    "AlgebraistGeneralKernel",
    "AlgebraistGeneratorCompiler",
    "AlgebraistGeneratorEmitter",
    "AlgebraistGeneratorGeneral",
    "AlgebraistGeneratorSpecialist",
    "AlgebraistKernel",
    "AlgebraistLayout",
    "AlgebraistLayoutBroadcast",
    "AlgebraistLayoutField",
    "AlgebraistLayoutLooped",
    "AlgebraistLayoutPath",
    "AlgebraistLayoutPolicy",
    "AlgebraistLayoutScalar",
    "AlgebraistLayoutUnravel",
    "AlgebraistRequest",
    "AlgebraistRuntimeGeneral",
    "AlgebraistRuntimeSpecialist",
    "AlgebraistSpecialist",
    "AlgebraistStencil",
    "AlgebraistAllocator",
]
