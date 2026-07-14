from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GeneratorMode = Literal["runtime", "generated"]
"""Whether a generator should build runtime providers or emitted kernels."""

GeneratorMutationStyle = Literal["in_place", "functional"]
"""Whether generated code mutates output objects or returns new values."""

GeneratorTraversalStyle = Literal[
    "scalar",
    "looped",
    "unrolled",
    "vectorized",
    "backend_kernel",
]
"""Preferred field traversal shape for generated code."""

GeneratorExpressionStyle = Literal[
    "python",
    "array_expression",
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

    mode: GeneratorMode = "runtime"
    mutation: GeneratorMutationStyle = "in_place"
    traversal: GeneratorTraversalStyle = "looped"
    expression: GeneratorExpressionStyle = "python"
    scalar: GeneratorScalarStyle = "python"

    @classmethod
    def runtime(cls) -> "GeneratorPolicy":
        """Use runtime providers when structure is not rich enough for codegen."""

        return cls(mode="runtime")

    @classmethod
    def mutable_looped(cls) -> "GeneratorPolicy":
        """Generate Python-compatible loops that mutate output objects."""

        return cls(
            mode="generated",
            mutation="in_place",
            traversal="looped",
            expression="python",
            scalar="python",
        )

    @classmethod
    def mutable_vectorized(cls) -> "GeneratorPolicy":
        """Generate whole-array expressions that mutate output objects."""

        return cls(
            mode="generated",
            mutation="in_place",
            traversal="vectorized",
            expression="array_expression",
            scalar="python",
        )

    @classmethod
    def functional_vectorized(cls) -> "GeneratorPolicy":
        """Generate whole-array expressions that return new values."""

        return cls(
            mode="generated",
            mutation="functional",
            traversal="vectorized",
            expression="array_expression",
            scalar="item",
        )

    @classmethod
    def backend_kernel(cls) -> "GeneratorPolicy":
        """Generate backend-native kernels instead of plain Python source."""

        return cls(
            mode="generated",
            mutation="in_place",
            traversal="backend_kernel",
            expression="backend_kernel",
            scalar="item",
        )


__all__ = [
    "GeneratorExpressionStyle",
    "GeneratorMode",
    "GeneratorMutationStyle",
    "GeneratorPolicy",
    "GeneratorScalarStyle",
    "GeneratorTraversalStyle",
]
