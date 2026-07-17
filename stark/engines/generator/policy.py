from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

GeneratorMutationStyle = Literal["in_place", "functional"]
"""Whether generated code mutates output objects or returns new values."""

GeneratorTraversalStyle = Literal[
    "scalar",
    "looped",
    "unrolled",
    "vectorized",
    "elementwise",
    "backend_kernel",
]
"""Preferred field traversal shape for generated code."""

GeneratorExpressionStyle = Literal[
    "python",
    "array_expression",
    "elementwise",
    "backend_kernel",
]
"""Preferred expression language emitted by generated code."""

GeneratorScalarStyle = Literal["python", "item"]
"""How backend scalar values should cross into Python control code."""


@dataclass(frozen=True, slots=True)
class GeneratorPolicy:
    """Code-shape decisions shared by generator families.

    A generator policy describes how source should be shaped before any
    accelerator compiles it. Engines own the concrete accelerator, carriers,
    allocator, and frame; this policy only records the choices that code
    generators need in order to emit compatible kernels.
    """

    mutation: GeneratorMutationStyle = "in_place"
    traversal: GeneratorTraversalStyle = "looped"
    expression: GeneratorExpressionStyle = "python"
    scalar: GeneratorScalarStyle = "python"

class GeneratorPolicyLike(Protocol):
    """Source-shape decisions understood by Generator operation families."""

    @property
    def mutation(self) -> GeneratorMutationStyle:
        ...

    @property
    def traversal(self) -> GeneratorTraversalStyle:
        ...

    @property
    def expression(self) -> GeneratorExpressionStyle:
        ...

    @property
    def scalar(self) -> GeneratorScalarStyle:
        ...


__all__ = [
    "GeneratorExpressionStyle",
    "GeneratorMutationStyle",
    "GeneratorPolicy",
    "GeneratorPolicyLike",
    "GeneratorScalarStyle",
    "GeneratorTraversalStyle",
]
