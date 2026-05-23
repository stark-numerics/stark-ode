from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar, ParamSpec, cast

from stark.accelerators.absent import AcceleratorAbsent
from stark.algebraist.arity import AlgebraistArity
from stark.algebraist.delta import AlgebraistDeltaKernel, AlgebraistDeltaStencil
from stark.algebraist.update import AlgebraistUpdateKernel, AlgebraistUpdateStencil
from stark.algebraist.workbench import AlgebraistWorkbench
from stark.contracts.acceleration import AcceleratorLike
from stark.contracts.translations import State, Translation

try:  # layout is optional context for runtime, but accepted for generate symmetry.
    from stark.algebraist.layout import AlgebraistLayout
except Exception:  # pragma: no cover - defensive during staged refactors
    AlgebraistLayout = object  # type: ignore[misc, assignment]


KernelType = TypeVar("KernelType")
StateType = TypeVar("StateType", bound=State)
TranslationType = TypeVar("TranslationType", bound=Translation)
RuntimeKernel = Callable[..., TranslationType]

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
                f"combine{self.arity} requires {self.arity} coefficient/translation pairs and out."
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
                f"combine{self.arity} requires {self.arity} coefficient/translation pairs and out."
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
        out: TranslationType,
        *translations: TranslationType,
    ) -> TranslationType:
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
class AlgebraistRuntimeUpdateKernel(Generic[StateType, TranslationType]):
    """Runtime fixed-coefficient state-update kernel."""

    delta: AlgebraistDeltaKernel[TranslationType]
    scratch: TranslationType

    def __call__(
        self,
        step: float,
        result: StateType,
        origin: StateType,
        *translations: TranslationType,
    ) -> StateType:
        delta = self.delta(step, self.scratch, *translations)
        delta(origin, result)
        return result


def _validate_coefficients(coefficients: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(coefficients)
    if not normalized:
        raise ValueError("Algebraist stencil must contain at least one coefficient.")
    return normalized


@dataclass(slots=True)
class AlgebraistRuntimeSupport(Generic[TranslationType]):
    """Shared implementation support for runtime Algebraist providers.

    Runtime providers construct this internally. It centralizes validation,
    direct-kernel resolution, fallback generation, synthesis, scratch allocation,
    and accelerator decoration.
    """

    translation: TranslationType
    workbench: AlgebraistWorkbench[TranslationType]
    layout: AlgebraistLayout | None = None
    linear_combine: Sequence[Callable[..., TranslationType]] | None = None
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)
    _fallback: AlgebraistRuntimeFallbackCombine[TranslationType] = field(init=False, repr=False)
    _kernels: dict[int, RuntimeKernel[TranslationType]] = field(init=False, repr=False)
    _has_direct_combine2: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.workbench, "allocate_translation", None)):
            raise TypeError("AlgebraistRuntimeSupport.workbench must provide allocate_translation().")

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
        return self.workbench.allocate_translation()

    def provide_general(self, request: AlgebraistArity) -> RuntimeKernel[TranslationType]:
        if request.value not in self._kernels:
            self._kernels[request.value] = self._build_general(request)
        return self._kernels[request.value]

    def provide_tuple(self, max_arity: int = 12) -> tuple[RuntimeKernel[TranslationType], ...]:
        if max_arity < 1:
            raise ValueError("max_arity must be at least 1.")
        return tuple(self.provide_general(AlgebraistArity(arity)) for arity in range(1, max_arity + 1))

    def provide_delta(self, request: AlgebraistDeltaStencil) -> AlgebraistDeltaKernel[TranslationType]:
        coefficients = _validate_coefficients(request.coefficients)
        combine = self.provide_general(AlgebraistArity(len(coefficients)))
        kernel = AlgebraistRuntimeDeltaKernel(
            scale=request.scale,
            coefficients=coefficients,
            combine=combine,
        )
        return cast(
            AlgebraistDeltaKernel[TranslationType],
            self.accelerate(kernel, label="runtime.delta", arity=len(coefficients)),
        )

    def provide_update(
        self,
        request: AlgebraistUpdateStencil,
    ) -> AlgebraistUpdateKernel[StateType, TranslationType]:
        coefficients = _validate_coefficients(request.coefficients)
        delta = self.provide_delta(request)
        scratch = self.allocate_translation()
        kernel = AlgebraistRuntimeUpdateKernel[StateType, TranslationType](
            delta=delta,
            scratch=scratch,
        )
        return cast(
            AlgebraistUpdateKernel[StateType, TranslationType],
            self.accelerate(kernel, label="runtime.update", arity=len(coefficients)),
        )

    def accelerate(
        self,
        kernel: KernelType,
        *,
        label: str,
        **values: object,
    ) -> KernelType:
        accelerated = self.accelerator.resolve_support(kernel, label=label, **values)
        return cast(KernelType, accelerated)

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
    "AlgebraistRuntimeCombineSynthesizer",
    "AlgebraistRuntimeDeltaKernel",
    "AlgebraistRuntimeFallbackCombine",
    "AlgebraistRuntimeFallbackKernel",
    "AlgebraistRuntimeSupport",
    "AlgebraistRuntimeUpdateKernel",
    "RuntimeKernel",
]
