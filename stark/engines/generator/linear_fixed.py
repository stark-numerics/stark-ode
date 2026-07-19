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
from stark.engines.generator.request import (
    GeneratorRequestApplyTranslationLike,
    GeneratorRequestLinearFixedLike,
)


@dataclass(slots=True)
class GeneratorLinearFixed(Generic[StateType, TranslationType]):
    """Generate or bind fixed-coefficient linear kernels."""

    frame: FrameLike
    accelerator: Accelerator = field(default_factory=AcceleratorNone)
    policy: GeneratorPolicyLike = field(default_factory=GeneratorPolicy)
    allocator: AllocatorLike[StateType, TranslationType] | None = None

    def __call__(
        self,
        request: GeneratorRequestLinearFixedLike,
    ) -> Callable[..., object]:
        return self.generated(request)

    def generated(self, request: GeneratorRequestLinearFixedLike) -> Callable[..., object]:
        source = GeneratorLinearFixedSource(self.frame, policy=self.policy)(request)
        return GeneratorCompiler(self.accelerator).compile(source)

    def apply_translation(
        self,
        request: GeneratorRequestApplyTranslationLike,
    ) -> Callable[..., object]:
        del request
        source = GeneratorLinearFixedSource(self.frame, policy=self.policy).unit_apply()
        return GeneratorCompiler(self.accelerator).compile(source)

__all__ = ["GeneratorLinearFixed"]
