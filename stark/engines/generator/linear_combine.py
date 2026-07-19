from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic

from stark.core.contracts.engines.accelerator import Accelerator
from stark.core.contracts.engines.allocator import AllocatorLike
from stark.core.contracts.problem.frame import FrameLike
from stark.core.contracts.problem.state import StateType
from stark.core.contracts.problem.translation import TranslationType
from stark.engines.accelerators.none import AcceleratorNone
from stark.engines.generator.compiler import GeneratorCompiler
from stark.engines.generator.linear_fixed_source import GeneratorLinearFixedSource
from stark.engines.generator.policy import GeneratorPolicy, GeneratorPolicyLike
from stark.engines.generator.request import GeneratorRequestLinearCombineLike


@dataclass(slots=True)
class GeneratorLinearCombine(Generic[StateType, TranslationType]):
    """Generate or bind runtime-coefficient linear-combination kernels."""

    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)
    allocator: AllocatorLike[StateType, TranslationType] | None = None

    def __call__(
        self,
        request: GeneratorRequestLinearCombineLike,
    ) -> Callable[..., object]:
        return self.generated(request)

    def generated(
        self,
        request: GeneratorRequestLinearCombineLike,
    ) -> Callable[..., object]:
        source = GeneratorLinearFixedSource(self.frame, policy=self.policy).emit(
            kind="general",
            coefficients=None,
            arity=self.arity(request),
        )
        return GeneratorCompiler(self.accelerator).compile(source)

    @staticmethod
    def arity(request: GeneratorRequestLinearCombineLike) -> int:
        arity = int(request.arity)
        if arity < 1:
            raise ValueError(f"linear_combine arity must be at least 1; got {arity}.")
        return arity


__all__ = ["GeneratorLinearCombine"]
