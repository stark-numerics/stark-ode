"""Contracts and type variables for translation objects.

Translation objects carry the linear update structure used by schemes,
resolvents, and inverters. They are separate from user state objects so STARK
can build weighted update combinations without needing to understand the
concrete state frame.
"""

from __future__ import annotations

from numbers import Real
from typing import Any, Protocol, Self, TypeVar

from stark.core.contracts.contract_audit import AuditRecorder


class Translation(Protocol):
    """
    A state update object carrying the linear structure of the problem.

    STARK separates nonlinear mutable state from linear translation objects.
    Schemes build weighted combinations of translations, and a translation can
    then be applied to a state to produce an updated state.

    A translation should behave like an element of the tangent space around a
    state. In practice that means:

    - `translation(origin, result)` applies the update to `origin` and writes
      the updated state into `result`
    - `norm()` measures the size of the update
    - `+` and scalar multiplication provide the linear operations STARK uses in
      explicit and implicit methods
    """

    def __call__(self, origin: Any, result: Any) -> None:
        ...

    def norm(self) -> float:
        ...

    def __add__(self, other: Self) -> Self:
        ...

    def __rmul__(self, scalar: float) -> Self:
        ...


TranslationType = TypeVar("TranslationType", bound=Translation)
"""Invariant type variable for protocols that both accept and return translations.

Use this when a contract must preserve the exact translation family across
input and output positions. Examples include coordinate bases, mutable block
containers, and helpers that receive a translation and return the same concrete
translation type.
"""

TranslationTypeCovariant = TypeVar(
    "TranslationTypeCovariant",
    bound=Translation,
    covariant=True,
)
"""Covariant type variable for producers of translation objects.

Use this when a contract only returns translations, such as allocator methods
that manufacture blank translation work buffers. A producer of a concrete
translation is safely usable wherever a broader translation producer is
expected.
"""

TranslationTypeContravariant = TypeVar(
    "TranslationTypeContravariant",
    bound=Translation,
    contravariant=True,
)
"""Contravariant type variable for consumers of translation objects.

Use this when a contract only accepts translations, such as derivative workers,
operators, inner products, and norms. Consumer protocols are contravariant so a
worker that can handle a broad translation type can also stand in for a worker
that is only required to handle a narrower one.
"""


class TranslationAudit:
    """Record checks for the translation algebra contract."""

    def __call__(
        self,
        recorder: AuditRecorder,
        translation: Any,
        *,
        exercise: bool = True,
        state: Any | None = None,
        result_state: Any | None = None,
        sample_translation: Any | None = None,
    ) -> None:
        recorder.check(callable(translation), "Translation is callable.", "Add __call__(origin, result) to the translation.")
        recorder.check(callable(getattr(translation, "norm", None)), "Translation provides norm().", "Add norm() returning a float.")
        recorder.check(
            callable(getattr(translation, "__add__", None)),
            "Translation provides __add__.",
            "Add __add__(other) for the fallback linear-combine path.",
        )
        recorder.check(
            callable(getattr(translation, "__rmul__", None)),
            "Translation provides __rmul__.",
            "Add __rmul__(scalar) for the fallback linear-combine path.",
        )

        linear_combine = getattr(translation, "linear_combine", None)
        if linear_combine is None:
            recorder.check(True, "Translation uses the generic __add__ / __rmul__ fallback.")
        elif not isinstance(linear_combine, (list, tuple)):
            recorder.check(
                False,
                "Translation.linear_combine is a list or tuple.",
                "Set linear_combine = [scale, combine2, ...].",
            )
        else:
            recorder.check(True, "Translation.linear_combine is present.")
            for index, combine in enumerate(linear_combine, start=1):
                arity_name = "scale" if index == 1 else f"combine{index}"
                recorder.check(
                    callable(combine),
                    f"Translation.linear_combine[{index - 1}] provides {arity_name}.",
                    f"Add a callable {arity_name} implementation at linear_combine[{index - 1}].",
                )

        if not exercise:
            return

        if state is not None and result_state is not None and callable(translation):
            try:
                translation(state, result_state)
            except Exception as exc:
                recorder.record_exception("Translation(origin, result) can be called.", exc)
            else:
                recorder.check(True, "Translation(origin, result) can be called.")

        norm = getattr(translation, "norm", None)
        if callable(norm):
            try:
                value = norm()
            except Exception as exc:
                recorder.record_exception("Translation.norm() succeeds.", exc)
            else:
                recorder.check(
                    isinstance(value, Real),
                    "Translation.norm() returns a real number.",
                    "Return a float-like norm from Translation.norm().",
                )

        if sample_translation is not None:
            self.exercise_linear_combine(recorder, translation, sample_translation)

    @staticmethod
    def exercise_linear_combine(recorder: AuditRecorder, translation: Any, sample_translation: Any) -> None:
        linear_combine = getattr(translation, "linear_combine", None)
        if not isinstance(linear_combine, (list, tuple)):
            return

        for index, combine in enumerate(linear_combine, start=1):
            if not callable(combine):
                continue
            args: list[Any] = []
            for term in range(index):
                args.extend([float(term + 1), sample_translation])
            args.append(sample_translation)
            arity_name = "scale" if index == 1 else f"combine{index}"
            try:
                combine(*args)
            except Exception as exc:
                recorder.record_exception(f"Translation {arity_name} can be called.", exc)
            else:
                recorder.check(True, f"Translation {arity_name} can be called.")

__all__ = [
    "Translation",
    "TranslationType",
    "TranslationTypeCovariant",
    "TranslationTypeContravariant",
    "TranslationAudit",
]
