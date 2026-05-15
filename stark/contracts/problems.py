from __future__ import annotations

"""
Protocols for problem-facing maths and state-allocation integration points.

This file intentionally groups the contracts a user most often supplies when
teaching STARK about a new problem:

- derivative workers defining the right-hand side
- optional linearizers for Newton-style implicit solves
- the workbench that allocates and copies the user's state objects

The first two are problem maths. The third is the memory/layout handshake that
lets the rest of the library work with arbitrary mutable state types.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from stark.contracts.audit_support import AuditRecorder
from stark.contracts.translations import Operator, State, Translation
from stark.interval import Interval


class Derivative(Protocol):
    def __call__(self, interval: Interval, state: Any, out: Any) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ImExDerivative:
    """
    Split right-hand side for implicit-explicit time stepping.

    IMEX methods treat one part of the derivative explicitly and one part
    implicitly. This small carrier keeps those two workers together in a
    single object that is easy to audit, document, and pass into a scheme.

    The full right-hand side is understood as

        f(t, x) = f_implicit(t, x) + f_explicit(t, x)

    where both parts write into translation objects in place.
    """

    implicit: Derivative
    explicit: Derivative

    @property
    def im(self) -> Derivative:
        return self.implicit

    @property
    def ex(self) -> Derivative:
        return self.explicit


class Linearizer(Protocol):
    """
    Fill `out` with the local Jacobian action of the derivative at `state`.

    This is the contract that asks the user to do some problem-specific maths.
    Given a nonlinear derivative

        x' = f(t, x),

    the linearizer must provide the action of the Jacobian

        J(t, state)[translation]

    as an `Operator`. STARK does not ask for a dense matrix. It asks for a
    callable linear operator that, given an input translation, writes the
    Jacobian image into `out`.
    """

    def __call__(self, interval: Interval, state: Any, out: Any) -> None:
        ...


class Workbench(Protocol):
    """
    Factory for reusable scratch objects and state-copy operations.

    This is the main integration point for user-defined state types. A custom
    workbench tells STARK how to:

    - allocate mutable state objects
    - copy one state into another
    - allocate translation objects compatible with that state

    Once this contract is satisfied, the built-in schemes, resolvents, and
    inverters can reuse those objects without knowing the concrete state shape.
    """

    def allocate_state(self) -> Any:
        ...

    def copy_state(self, dst: Any, src: Any) -> None:
        ...

    def allocate_translation(self) -> Any:
        ...


class ProblemAudit:
    class OperatorProbe:
        __slots__ = ("apply",)

        def __init__(self) -> None:
            self.apply = self._unset

        @staticmethod
        def _unset(translation: Any, out: Any) -> None:
            del translation, out
            raise RuntimeError("Linearizer did not configure the operator probe.")

        def __call__(self, translation: Any, out: Any) -> Any:
            return self.apply(translation, out)

    @staticmethod
    def derivative(recorder: AuditRecorder, derivative: Any) -> None:
        recorder.check(
            callable(derivative),
            "Derivative is callable.",
            "Provide a callable derivative(interval, state, translation).",
        )

    @staticmethod
    def imex_derivative(recorder: AuditRecorder, imex_derivative: Any) -> None:
        implicit = getattr(imex_derivative, "implicit", None)
        explicit = getattr(imex_derivative, "explicit", None)
        recorder.check(
            callable(implicit),
            "ImExDerivative provides implicit(interval, state, translation).",
            "Set imex_derivative.implicit to a callable derivative worker.",
        )
        recorder.check(
            callable(explicit),
            "ImExDerivative provides explicit(interval, state, translation).",
            "Set imex_derivative.explicit to a callable derivative worker.",
        )

    @staticmethod
    def workbench(
        recorder: AuditRecorder,
        workbench: Any,
        *,
        exercise: bool = True,
    ) -> tuple[Any | None, Any | None, Any | None]:
        allocate_state = getattr(workbench, "allocate_state", None)
        copy_state = getattr(workbench, "copy_state", None)
        allocate_translation = getattr(workbench, "allocate_translation", None)

        recorder.check(callable(allocate_state), "Workbench provides allocate_state().", "Add allocate_state() returning a blank mutable state.")
        recorder.check(callable(copy_state), "Workbench provides copy_state(dst, src).", "Add copy_state(dst, src) to support safe updates and snapshots.")
        recorder.check(callable(allocate_translation), "Workbench provides allocate_translation().", "Add allocate_translation() returning a blank translation.")

        if not exercise:
            return None, None, None

        sample_state = None
        second_state = None
        sample_translation = None

        if callable(allocate_state):
            try:
                sample_state = allocate_state()
                second_state = allocate_state()
            except Exception as exc:
                recorder.record_exception("Workbench.allocate_state() succeeds.", exc)
            else:
                recorder.check(True, "Workbench.allocate_state() succeeds.")

        if callable(allocate_translation):
            try:
                sample_translation = allocate_translation()
            except Exception as exc:
                recorder.record_exception("Workbench.allocate_translation() succeeds.", exc)
            else:
                recorder.check(True, "Workbench.allocate_translation() succeeds.")

        return sample_state, second_state, sample_translation

    @staticmethod
    def linearizer(recorder: AuditRecorder, linearizer: Any) -> None:
        recorder.check(
            callable(linearizer),
            "Linearizer provides __call__(interval, state, out).",
            "Provide a callable linearizer(interval, state, out).",
        )

    @staticmethod
    def exercise_derivative(
        recorder: AuditRecorder,
        derivative: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        try:
            derivative(interval, state, translation)
        except Exception as exc:
            recorder.record_exception("Derivative(interval, state, translation) can be called.", exc)
        else:
            recorder.check(True, "Derivative(interval, state, translation) can be called.")

    @staticmethod
    def exercise_imex_derivative(
        recorder: AuditRecorder,
        imex_derivative: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        implicit = getattr(imex_derivative, "implicit", None)
        explicit = getattr(imex_derivative, "explicit", None)
        if callable(implicit):
            try:
                implicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("ImExDerivative implicit part can be called.", exc)
            else:
                recorder.check(True, "ImExDerivative implicit part can be called.")
        if callable(explicit):
            try:
                explicit(interval, state, translation)
            except Exception as exc:
                recorder.record_exception("ImExDerivative explicit part can be called.", exc)
            else:
                recorder.check(True, "ImExDerivative explicit part can be called.")

    @staticmethod
    def exercise_copy_state(recorder: AuditRecorder, workbench: Any, dst: Any, src: Any) -> None:
        copy_state = getattr(workbench, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(dst, src)
        except Exception as exc:
            recorder.record_exception("Workbench.copy_state(dst, src) can copy a provided state.", exc)
        else:
            recorder.check(True, "Workbench.copy_state(dst, src) can copy a provided state.")

    @staticmethod
    def exercise_workbench_copy(recorder: AuditRecorder, workbench: Any, dst: Any, src: Any) -> None:
        copy_state = getattr(workbench, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(dst, src)
        except Exception as exc:
            recorder.record_exception("Workbench.copy_state(dst, src) succeeds on blank states.", exc)
        else:
            recorder.check(True, "Workbench.copy_state(dst, src) succeeds on blank states.")

    @staticmethod
    def exercise_linearizer(
        recorder: AuditRecorder,
        linearizer: Any,
        interval: Any,
        state: Any,
        translation: Any,
    ) -> None:
        operator = ProblemAudit.OperatorProbe()
        try:
            linearizer(interval, state, operator)
        except Exception as exc:
            recorder.record_exception("Linearizer(interval, state, out) can be called.", exc)
            return
        else:
            recorder.check(True, "Linearizer(interval, state, out) can be called.")

        try:
            result = operator(translation, translation)
        except Exception as exc:
            recorder.record_exception("Linearizer configures an operator callable.", exc)
            return
        else:
            recorder.check(True, "Linearizer configures an operator callable.")
            recorder.check(
                result is None,
                "Linearizer-configured operators fill in place and return None.",
                "Make operator(translation, out) mutate out and return None.",
            )


__all__ = [
    "Derivative",
    "ImExDerivative",
    "Linearizer",
    "ProblemAudit",
    "Workbench",
]





