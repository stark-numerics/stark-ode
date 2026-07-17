from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, overload


class GeneratorRequestLike(Protocol):
    """Minimal request contract used by Generator dispatch."""

    @property
    def operation(self) -> str:
        ...


class GeneratorRequestLinearFixedLike(GeneratorRequestLike, Protocol):
    """Fixed-coefficient linear request."""

    @property
    def operation(self) -> Literal["linear_fixed"]:
        ...

    @property
    def coefficients(self) -> Sequence[float]:
        ...

    @property
    def scale(self) -> float:
        ...

    @property
    def apply(self) -> bool:
        ...


class GeneratorRequestApplyTranslationLike(GeneratorRequestLike, Protocol):
    """Request a kernel applying one translation to one origin state."""

    @property
    def operation(self) -> Literal["apply_translation"]:
        ...


@dataclass(frozen=True, slots=True)
class GeneratorRequestApplyTranslation:
    """Request `result = origin + translation` for a frame-backed state."""

    operation: Literal["apply_translation"] = "apply_translation"


class GeneratorRequestLinearCombineLike(GeneratorRequestLike, Protocol):
    """Runtime-coefficient linear-combination request."""

    @property
    def operation(self) -> Literal["linear_combine"]:
        ...

    @property
    def arity(self) -> int:
        ...


class GeneratorRequestLinearCombineTableLike(GeneratorRequestLike, Protocol):
    """Request a family of linear-combination kernels up to one arity."""

    @property
    def operation(self) -> Literal["linear_combine_table"]:
        ...

    @property
    def max_arity(self) -> int:
        ...


@dataclass(frozen=True, slots=True)
class GeneratorRequestLinearCombine:
    """Request one runtime-coefficient linear-combination kernel."""

    arity: int
    operation: Literal["linear_combine"] = "linear_combine"


@dataclass(frozen=True, slots=True)
class GeneratorRequestLinearCombineTable:
    """Request linear-combination kernels for arities `1..max_arity`."""

    max_arity: int
    operation: Literal["linear_combine_table"] = "linear_combine_table"

    def __post_init__(self) -> None:
        if self.max_arity < 1:
            raise ValueError(
                f"linear_combine_table max_arity must be at least 1; got {self.max_arity}."
            )


class GeneratorRequestNormLike(GeneratorRequestLike, Protocol):
    """Frame norm request."""

    @property
    def operation(self) -> Literal["norm"]:
        ...

    @property
    def kind(self) -> str:
        ...


class GeneratorRequestInnerProductLike(GeneratorRequestLike, Protocol):
    """Frame inner-product request."""

    @property
    def operation(self) -> Literal["inner_product"]:
        ...

    @property
    def kind(self) -> str:
        ...


@dataclass(frozen=True, slots=True)
class GeneratorRequestNorm:
    """Request a frame-aware norm kernel."""

    kind: str = "default"
    operation: Literal["norm"] = "norm"


@dataclass(frozen=True, slots=True)
class GeneratorRequestInnerProduct:
    """Request a frame-aware inner-product kernel."""

    kind: str = "default"
    operation: Literal["inner_product"] = "inner_product"


class GeneratorLike(Protocol):
    """Callable request dispatcher for generated engine hooks."""

    @overload
    def __call__(
        self,
        request: GeneratorRequestApplyTranslationLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearCombineLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearCombineTableLike,
    ) -> tuple[Callable[..., object], ...]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLinearFixedLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestNormLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestInnerProductLike,
    ) -> Callable[..., object]:
        ...

    @overload
    def __call__(
        self,
        request: GeneratorRequestLike,
    ) -> Callable[..., object] | tuple[Callable[..., object], ...]:
        ...

    def __call__(
        self,
        request: GeneratorRequestLike,
    ) -> Callable[..., object] | tuple[Callable[..., object], ...]:
        ...


__all__ = [
    "GeneratorLike",
    "GeneratorRequestApplyTranslation",
    "GeneratorRequestApplyTranslationLike",
    "GeneratorRequestInnerProduct",
    "GeneratorRequestInnerProductLike",
    "GeneratorRequestLike",
    "GeneratorRequestLinearCombine",
    "GeneratorRequestLinearCombineLike",
    "GeneratorRequestLinearCombineTable",
    "GeneratorRequestLinearCombineTableLike",
    "GeneratorRequestLinearFixedLike",
    "GeneratorRequestNorm",
    "GeneratorRequestNormLike",
]
