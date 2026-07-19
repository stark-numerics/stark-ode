from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, cast

from stark.core.contracts.problem.translation import TranslationType, TranslationTypeCovariant

AllocatorRuntimeKernel = Callable[..., TranslationType]


class LinearCombineScratchAllocatorLike(Protocol[TranslationTypeCovariant]):
    """Allocator shape needed to synthesize missing runtime combine kernels."""

    def allocate_translation(self) -> TranslationTypeCovariant:
        ...


@dataclass(frozen=True, slots=True)
class AllocatorRuntimeLinearCombineFallback(Generic[TranslationType]):
    """Return-style linear combine using translation arithmetic."""

    arity: int

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError("Fallback combine arity must be at least 1.")

    def __call__(self, *terms: object) -> TranslationType:
        first_coefficient = cast(float, terms[0])
        first_translation = cast(TranslationType, terms[1])
        result = first_coefficient * first_translation

        for index in range(2, len(terms) - 1, 2):
            coefficient = cast(float, terms[index])
            translation = cast(TranslationType, terms[index + 1])
            result = result + coefficient * translation

        return cast(TranslationType, result)


@dataclass(frozen=True, slots=True)
class AllocatorRuntimeLinearCombineSynthesizer(Generic[TranslationType]):
    """Balanced into-style synthesizer for missing higher-arity combines."""

    arity: int
    left_arity: int
    right_arity: int
    left_kernel: AllocatorRuntimeKernel[TranslationType]
    right_kernel: AllocatorRuntimeKernel[TranslationType]
    combine2: AllocatorRuntimeKernel[TranslationType]
    left: TranslationType
    right: TranslationType

    def __post_init__(self) -> None:
        if self.arity < 3:
            raise ValueError("Synthesized combine arity must be at least 3.")
        if self.left_arity + self.right_arity != self.arity:
            raise ValueError("Synthesized combine split must sum to the target arity.")

    def __call__(self, *terms: object) -> TranslationType:
        out = cast(TranslationType, terms[-1])
        paired_terms = terms[:-1]
        split = 2 * self.left_arity
        left_value = self.left_kernel(*paired_terms[:split], self.left)
        right_value = self.right_kernel(*paired_terms[split:], self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


@dataclass(frozen=True, slots=True)
class AllocatorRuntimeLinearCombine(Generic[TranslationType]):
    """Resolve direct, synthesized, or fallback runtime combine kernels."""

    allocator: LinearCombineScratchAllocatorLike[TranslationType] | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] = ()

    def table(self, max_arity: int) -> tuple[AllocatorRuntimeKernel[TranslationType], ...]:
        if max_arity < 1:
            raise ValueError(
                f"Linear-combine table requires max_arity >= 1; got {max_arity}."
            )
        return tuple(self.combine_for_arity(arity) for arity in range(1, max_arity + 1))

    def combine_for_arity(self, arity: int) -> AllocatorRuntimeKernel[TranslationType]:
        if arity < 1:
            arity = 1
        direct = self.direct_kernels_for_provider()
        return self.build_combine(arity=arity, direct=direct)

    def combine_for(
        self,
        out: TranslationType,
        arity: int,
    ) -> AllocatorRuntimeKernel[TranslationType]:
        if arity < 1:
            arity = 1
        direct = self.direct_kernels(out)
        return self.build_combine(arity=arity, direct=direct)

    def direct_kernels_for_provider(self) -> tuple[AllocatorRuntimeKernel[TranslationType], ...]:
        if self.linear_combine:
            return self.validate_kernels(self.linear_combine)
        if self.allocator is None:
            return ()
        return self.direct_kernels(self.allocator.allocate_translation())

    def direct_kernels(
        self,
        out: TranslationType,
    ) -> tuple[AllocatorRuntimeKernel[TranslationType], ...]:
        raw = getattr(out, "linear_combine", None)
        if raw is None:
            return ()
        return self.validate_kernels(raw)

    @staticmethod
    def validate_kernels(
        raw: Sequence[Callable[..., TranslationType]],
    ) -> tuple[AllocatorRuntimeKernel[TranslationType], ...]:
        if not isinstance(raw, (list, tuple)):
            raise TypeError("linear_combine must be a list or tuple of callables.")

        kernels: list[AllocatorRuntimeKernel[TranslationType]] = []
        for index, kernel in enumerate(raw):
            if not callable(kernel):
                name = "scale" if index == 0 else f"combine{index + 1}"
                raise TypeError(f"linear_combine[{index}] must be a callable {name}.")
            kernels.append(cast(AllocatorRuntimeKernel[TranslationType], kernel))
        return tuple(kernels)

    def build_combine(
        self,
        *,
        arity: int,
        direct: tuple[AllocatorRuntimeKernel[TranslationType], ...],
    ) -> AllocatorRuntimeKernel[TranslationType]:
        if len(direct) >= arity:
            return direct[arity - 1]
        if arity < 3 or len(direct) < 2 or self.allocator is None:
            return AllocatorRuntimeLinearCombineFallback[TranslationType](arity)

        left_arity = arity // 2
        right_arity = arity - left_arity
        return AllocatorRuntimeLinearCombineSynthesizer(
            arity=arity,
            left_arity=left_arity,
            right_arity=right_arity,
            left_kernel=self.build_combine(arity=left_arity, direct=direct),
            right_kernel=self.build_combine(arity=right_arity, direct=direct),
            combine2=direct[1],
            left=self.allocator.allocate_translation(),
            right=self.allocator.allocate_translation(),
        )


__all__ = [
    "AllocatorRuntimeKernel",
    "AllocatorRuntimeLinearCombine",
    "LinearCombineScratchAllocatorLike",
    "AllocatorRuntimeLinearCombineFallback",
    "AllocatorRuntimeLinearCombineSynthesizer",
]
