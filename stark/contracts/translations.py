from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Real
from typing import Any, Protocol, Self

from stark.contracts.audit_support import AuditRecorder


State = Any


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


@dataclass(slots=True)
class Block:
    """
    A grouped solver-space object built from one or more translations.

    Implicit schemes, resolvents, and inverters work in a product space of
    translations rather than on single translations alone. A one-stage implicit
    method therefore uses a one-item block, while multi-stage methods and
    quasi-Newton histories can use larger blocks.
    """

    items: list[Translation]

    def __repr__(self) -> str:
        return f"Block(size={len(self.items)!r})"

    def __str__(self) -> str:
        return f"block[{len(self.items)}]"

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> Translation:
        return self.items[index]

    def norm(self) -> float:
        if not self.items:
            return 0.0
        return sqrt(sum(item.norm() ** 2 for item in self.items))


class Operator(Protocol):
    """
    Fill `out` with the image of a translation under a linear operator.

    Users normally provide operators through a `Linearizer`. The operator does
    not need to expose a dense matrix. It only needs to apply the local linear
    map to a translation, which is enough for matrix-free inverters such as
    GMRES, FGMRES, and BiCGStab.
    """

    def __call__(self, translation: Translation, out: Translation) -> None:
        ...


class InnerProduct(Protocol):
    """
    Return the inner product of two translations.

    Norms alone are not enough for Krylov methods. If a resolvent or inverter
    needs orthogonalization or secant projections, the user must also provide
    an inner product compatible with the translation space.
    """

    def __call__(self, left: Any, right: Any) -> float:
        ...


class TranslationAudit:
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
            args: list[Any] = [sample_translation]
            for term in range(index):
                args.extend([float(term + 1), sample_translation])
            arity_name = "scale" if index == 1 else f"combine{index}"
            try:
                combine(*args)
            except Exception as exc:
                recorder.record_exception(f"Translation {arity_name} can be called.", exc)
            else:
                recorder.check(True, f"Translation {arity_name} can be called.")


__all__ = [
    "TranslationAudit",
    "Block",
    "InnerProduct",
    "Operator",
    "State",
    "Translation",
]





