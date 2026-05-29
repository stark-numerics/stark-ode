from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, cast

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.stencil import AlgebraistStencil
from stark.algebraist.allocator import AlgebraistAllocator
from stark.contracts.acceleration import AcceleratorLike
from stark.contracts.states import State
from stark.contracts.translations import Translation

try:  # layout is optional context for runtime, but accepted for generator symmetry.
    from stark.algebraist.layout import AlgebraistLayout
except Exception:  # pragma: no cover - defensive during staged refactors
    AlgebraistLayout = object  # type: ignore[misc, assignment]

StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)
RuntimeKernel = Callable[..., TranslationType]
AnyRuntimeKernel = TypeVar("AnyRuntimeKernel", bound=Callable[..., object])


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeFallbackKernel(Generic[TranslationType]):
    """Return-style fallback kernel using Translation's guaranteed algebra."""

    arity: int

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError("Fallback combine arity must be at least 1.")

    def __call__(self, *terms: object) -> TranslationType:
        expected = 2 * self.arity + 1
        if len(terms) != expected:
            raise TypeError(
                f"combine{self.arity} requires {self.arity} "
                "coefficient/translation pairs and out."
            )

        # The fallback path relies on Translation.__rmul__ and Translation.__add__.
        # It accepts `out` for compatibility with into-style combine call sites but
        # returns a fresh Translation because the base Translation contract has no
        # in-place linear-combine primitive.
        out = terms[-1]
        del out

        first_coefficient = cast(float, terms[0])
        first_translation = cast(TranslationType, terms[1])
        result = first_coefficient * first_translation

        for index in range(2, len(terms) - 1, 2):
            coefficient = cast(float, terms[index])
            translation = cast(TranslationType, terms[index + 1])
            result = result + coefficient * translation

        return cast(TranslationType, result)


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeFallbackCombine(Generic[TranslationType]):
    """Fallback provider for arbitrary arity when no useful combine2 exists."""

    def provide(self, request: AlgebraistArity) -> RuntimeKernel[TranslationType]:
        return AlgebraistRuntimeFallbackKernel[TranslationType](request.value)


@dataclass(frozen=True, slots=True)
class AlgebraistRuntimeCombineSynthesizer(Generic[TranslationType]):
    """Balanced into-style synthesizer for missing higher-arity combines."""

    arity: int
    left_arity: int
    right_arity: int
    left_kernel: RuntimeKernel[TranslationType]
    right_kernel: RuntimeKernel[TranslationType]
    combine2: RuntimeKernel[TranslationType]
    left: TranslationType
    right: TranslationType

    def __post_init__(self) -> None:
        if self.arity < 3:
            raise ValueError("Synthesized combine arity must be at least 3.")
        if self.left_arity + self.right_arity != self.arity:
            raise ValueError("Synthesized combine split must sum to the target arity.")

    def __call__(self, *terms: object) -> TranslationType:
        expected = 2 * self.arity + 1
        if len(terms) != expected:
            raise TypeError(
                f"combine{self.arity} requires {self.arity} "
                "coefficient/translation pairs and out."
            )

        out = cast(TranslationType, terms[-1])
        paired_terms = terms[:-1]
        split = 2 * self.left_arity

        left_value = self.left_kernel(*paired_terms[:split], self.left)
        right_value = self.right_kernel(*paired_terms[split:], self.right)
        return self.combine2(1.0, left_value, 1.0, right_value, out)


@dataclass(slots=True)
class AlgebraistRuntimeDeltaKernel(Generic[TranslationType]):
    """Runtime fixed-coefficient delta kernel."""

    scale: float
    coefficients: tuple[float, ...]
    combine: RuntimeKernel[TranslationType]

    def __call__(
        self,
        step: float,
        *terms: TranslationType,
    ) -> TranslationType:
        if len(terms) != len(self.coefficients) + 1:
            received = max(0, len(terms) - 1)
            raise TypeError(
                f"Delta kernel requires {len(self.coefficients)} translations; "
                f"received {received}."
            )

        translations = terms[:-1]
        out = terms[-1]

        if len(translations) != len(self.coefficients):
            raise TypeError(
                f"Delta kernel requires {len(self.coefficients)} translations; "
                f"received {len(translations)}."
            )

        effective_scale = step * self.scale
        terms: list[object] = []
        for coefficient, translation in zip(self.coefficients, translations):
            terms.append(effective_scale * coefficient)
            terms.append(translation)
        terms.append(out)
        return self.combine(*terms)


@dataclass(slots=True)
class AlgebraistRuntimeZeroDeltaKernel(Generic[TranslationType]):
    """Runtime delta kernel for empty fixed-coefficient stencils."""

    combine: RuntimeKernel[TranslationType]

    def __call__(
        self,
        step: float,
        out: TranslationType,
    ) -> TranslationType:
        del step
        return self.combine(0.0, out, out)


@dataclass(slots=True)
class AlgebraistRuntimeApplyKernel(Generic[StateType, TranslationType]):
    """Runtime fixed-coefficient state-apply kernel."""

    delta: Callable[..., TranslationType]
    scratch: TranslationType

    def __call__(
        self,
        step: float,
        origin: StateType,
        *terms: object,
    ) -> StateType:
        if not terms:
            raise TypeError("Apply kernel requires a result state.")

        translations = cast(tuple[TranslationType, ...], terms[:-1])
        result = cast(StateType, terms[-1])

        delta = self.delta(step, *translations, self.scratch)
        delta(origin, result)
        return result


def _validate_coefficients(coefficients: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(coefficient) for coefficient in coefficients)


@dataclass(slots=True)
class AlgebraistRuntimeSupport(Generic[TranslationType]):
    """Shared implementation support for runtime Algebraist providers."""

    translation: TranslationType
    allocator: AlgebraistAllocator[TranslationType]
    layout: AlgebraistLayout | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)
    _fallback: AlgebraistRuntimeFallbackCombine[TranslationType] = field(init=False, repr=False)
    _kernels: dict[int, RuntimeKernel[TranslationType]] = field(init=False, repr=False)
    _has_direct_combine2: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.allocator, "allocate_translation", None)):
            raise TypeError("AlgebraistRuntimeSupport.allocator must provide allocate_translation().")

        self._fallback = AlgebraistRuntimeFallbackCombine[TranslationType]()
        direct = self._validated_linear_combine()
        self._has_direct_combine2 = len(direct) >= 2
        self._kernels = {}

        for index, kernel in enumerate(direct, start=1):
            self._kernels[index] = self.accelerate(
                kernel,
                label="runtime.general.direct",
                arity=index,
            )

    def allocate_translation(self) -> TranslationType:
        return self.allocator.allocate_translation()

    def provide_general(self, request: AlgebraistArity) -> RuntimeKernel[TranslationType]:
        if request.value not in self._kernels:
            self._kernels[request.value] = self._build_general(request)
        return self._kernels[request.value]

    def provide_tuple(self, max_arity: int = 12) -> tuple[RuntimeKernel[TranslationType], ...]:
        if max_arity < 1:
            raise ValueError("max_arity must be at least 1.")
        return tuple(
            self.provide_general(AlgebraistArity(arity))
            for arity in range(1, max_arity + 1)
        )

    def provide_specialist(self, request: AlgebraistStencil) -> Callable[..., object]:
        coefficients = _validate_coefficients(request.coefficients)
        if request.apply:
            return self.provide_apply(request=request, coefficients=coefficients)
        return self.provide_delta(request=request, coefficients=coefficients)

    def provide_delta(
        self,
        *,
        request: AlgebraistStencil,
        coefficients: tuple[float, ...] | None = None,
    ) -> Callable[..., TranslationType]:
        if coefficients is None:
            coefficients = _validate_coefficients(request.coefficients)
        if not coefficients:
            combine = self.provide_general(AlgebraistArity(1))
            kernel = AlgebraistRuntimeZeroDeltaKernel(combine=combine)
            return self.accelerate(kernel, label="runtime.delta.zero", arity=0)
        combine = self.provide_general(AlgebraistArity(len(coefficients)))
        kernel = AlgebraistRuntimeDeltaKernel(
            scale=float(request.scale),
            coefficients=coefficients,
            combine=combine,
        )
        return self.accelerate(kernel, label="runtime.delta", arity=len(coefficients))

    def provide_apply(
        self,
        *,
        request: AlgebraistStencil,
        coefficients: tuple[float, ...] | None = None,
    ) -> Callable[..., StateType]:
        if coefficients is None:
            coefficients = _validate_coefficients(request.coefficients)
        delta = self.provide_delta(request=request, coefficients=coefficients)
        scratch = self.allocate_translation()
        kernel = AlgebraistRuntimeApplyKernel[StateType, TranslationType](
            delta=delta,
            scratch=scratch,
        )
        return self.accelerate(kernel, label="runtime.apply", arity=len(coefficients))

    def accelerate(
        self,
        kernel: AnyRuntimeKernel,
        *,
        label: str,
        **values: object,
    ) -> AnyRuntimeKernel:
        del label, values
        return kernel

    def _validated_linear_combine(self) -> tuple[RuntimeKernel[TranslationType], ...]:
        raw = self.linear_combine
        if raw is None:
            raw = getattr(self.translation, "linear_combine", None)
        if raw is None:
            return ()
        if not isinstance(raw, (list, tuple)):
            raise TypeError("linear_combine must be a list or tuple of callables.")

        kernels: list[RuntimeKernel[TranslationType]] = []
        for index, kernel in enumerate(raw):
            if not callable(kernel):
                name = "scale" if index == 0 else f"combine{index + 1}"
                raise TypeError(f"linear_combine[{index}] must be a callable {name}.")
            kernels.append(cast(RuntimeKernel[TranslationType], kernel))
        return tuple(kernels)

    def _build_general(self, request: AlgebraistArity) -> RuntimeKernel[TranslationType]:
        arity = request.value
        if not self._has_direct_combine2:
            return self.accelerate(
                self._fallback.provide(request),
                label="runtime.general.fallback",
                arity=arity,
            )

        if arity == 1 or arity == 2:
            return self.accelerate(
                self._fallback.provide(request),
                label="runtime.general.fallback",
                arity=arity,
            )

        left_arity = arity // 2
        right_arity = arity - left_arity
        kernel = AlgebraistRuntimeCombineSynthesizer(
            arity=arity,
            left_arity=left_arity,
            right_arity=right_arity,
            left_kernel=self.provide_general(AlgebraistArity(left_arity)),
            right_kernel=self.provide_general(AlgebraistArity(right_arity)),
            combine2=self.provide_general(AlgebraistArity(2)),
            left=self.allocate_translation(),
            right=self.allocate_translation(),
        )
        return self.accelerate(
            kernel,
            label="runtime.general.synthesized",
            arity=arity,
            left_arity=left_arity,
            right_arity=right_arity,
        )


__all__ = [
    "AlgebraistRuntimeApplyKernel",
    "AlgebraistRuntimeCombineSynthesizer",
    "AlgebraistRuntimeDeltaKernel",
    "AlgebraistRuntimeFallbackCombine",
    "AlgebraistRuntimeFallbackKernel",
    "AlgebraistRuntimeSupport",
    "AlgebraistRuntimeZeroDeltaKernel",
    "RuntimeKernel",
]
