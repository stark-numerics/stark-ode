from __future__ import annotations

from numbers import Real
from typing import Any


class AuditError(TypeError):
    """Raised when a STARK contract audit fails."""


class Auditor:
    """
    Check whether user-supplied objects satisfy the STARK contracts.

    Instantiate `Auditor(...)` to produce a checklist-style report before
    starting an integration. Internal STARK constructors use the `require_*`
    class methods to fail fast with helpful errors.
    """

    __slots__ = ("checks",)

    def __init__(
        self,
        *,
        state: Any | None = None,
        derivative: Any | None = None,
        translation: Any | None = None,
        workbench: Any | None = None,
        interval: Any | None = None,
        marcher: Any | None = None,
        scheme: Any | None = None,
        tolerance: Any | None = None,
        snapshots: bool = True,
        exercise: bool = True,
    ) -> None:
        self.checks: list[tuple[bool, str, str | None]] = []

        sample_state = None
        second_state = None
        sample_translation = None

        if derivative is not None:
            self._check(callable(derivative), "Derivative is callable.", "Provide a callable derivative(state, translation).")

        if translation is not None:
            self._audit_translation(translation)

        if workbench is not None:
            sample_state, second_state, sample_translation = self._audit_workbench(workbench, exercise)

        if tolerance is not None:
            self._audit_tolerance(tolerance)

        if scheme is not None:
            self._audit_scheme(scheme)

        if marcher is not None:
            self._audit_marcher(marcher, snapshots=snapshots)


        if interval is not None:
            self._audit_interval(interval, exercise=exercise)

        if exercise and workbench is not None and derivative is not None and state is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self._exercise_derivative(derivative, state, candidate_translation)

        if exercise and workbench is not None and state is not None:
            candidate_state = sample_state
            if candidate_state is not None:
                self._exercise_copy_state(workbench, candidate_state, state)

        if exercise and translation is not None and state is not None and sample_state is not None:
            self._exercise_translation_apply(translation, state, sample_state)

        if exercise and translation is not None:
            candidate_translation = sample_translation if sample_translation is not None else translation
            if candidate_translation is not None:
                self._exercise_linear_combine(translation, candidate_translation)

        if exercise and sample_state is not None and second_state is not None and workbench is not None:
            self._exercise_workbench_copy(workbench, sample_state, second_state)

    @property
    def ok(self) -> bool:
        return all(passed for passed, _summary, _detail in self.checks)

    def __repr__(self) -> str:
        passed = sum(1 for ok, _summary, _detail in self.checks if ok)
        total = len(self.checks)
        status = "ready" if self.ok else "incomplete"
        return f"Auditor(status={status!r}, passed={passed}, total={total})"

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise AuditError(str(self))

    def _check(self, passed: bool, summary: str, detail: str | None = None) -> None:
        self.checks.append((passed, summary, detail))

    def _record_exception(self, summary: str, exc: Exception, detail: str | None = None) -> None:
        message = detail if detail is not None else f"{type(exc).__name__}: {exc}"
        self._check(False, summary, message)

    def _audit_translation(self, translation: Any) -> None:
        self._check(callable(translation), "Translation is callable.", "Add __call__(origin, result) to the translation.")
        self._check(callable(getattr(translation, "norm", None)), "Translation provides norm().", "Add norm() returning a float.")
        self._check(callable(getattr(translation, "__add__", None)), "Translation provides __add__.", "Add __add__(other) for the fallback linear-combine path.")
        self._check(callable(getattr(translation, "__rmul__", None)), "Translation provides __rmul__.", "Add __rmul__(scalar) for the fallback linear-combine path.")

        linear_combine = getattr(translation, "linear_combine", None)
        if linear_combine is None:
            self._check(True, "Translation uses the generic __add__ / __rmul__ fallback.")
            return

        if not isinstance(linear_combine, (list, tuple)):
            self._check(False, "Translation.linear_combine is a list or tuple.", "Set linear_combine = [scale, combine2, ...].")
            return

        self._check(True, "Translation.linear_combine is present.")
        for index, combine in enumerate(linear_combine, start=1):
            arity_name = "scale" if index == 1 else f"combine{index}"
            self._check(
                callable(combine),
                f"Translation.linear_combine[{index - 1}] provides {arity_name}.",
                f"Add a callable {arity_name} implementation at linear_combine[{index - 1}].",
            )

    def _audit_workbench(self, workbench: Any, exercise: bool) -> tuple[Any | None, Any | None, Any | None]:
        allocate_state = getattr(workbench, "allocate_state", None)
        copy_state = getattr(workbench, "copy_state", None)
        allocate_translation = getattr(workbench, "allocate_translation", None)

        self._check(callable(allocate_state), "Workbench provides allocate_state().", "Add allocate_state() returning a blank mutable state.")
        self._check(callable(copy_state), "Workbench provides copy_state(dst, src).", "Add copy_state(dst, src) to support safe updates and snapshots.")
        self._check(callable(allocate_translation), "Workbench provides allocate_translation().", "Add allocate_translation() returning a blank translation.")

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
                self._record_exception("Workbench.allocate_state() succeeds.", exc)
            else:
                self._check(True, "Workbench.allocate_state() succeeds.")

        if callable(allocate_translation):
            try:
                sample_translation = allocate_translation()
            except Exception as exc:
                self._record_exception("Workbench.allocate_translation() succeeds.", exc)
            else:
                self._check(True, "Workbench.allocate_translation() succeeds.")

        return sample_state, second_state, sample_translation

    def _audit_interval(self, interval: Any, exercise: bool) -> None:
        for name in ("present", "step", "stop"):
            self._check(hasattr(interval, name), f"Interval provides {name}.", f"Add a {name} attribute to the interval.")

        increment = getattr(interval, "increment", None)
        copy = getattr(interval, "copy", None)
        self._check(callable(increment), "Interval provides increment(dt).", "Add increment(dt) to advance present by the accepted step.")
        self._check(callable(copy), "Interval provides copy().", "Add copy() so snapshot integration can clone the interval.")

        if not exercise:
            return

        if callable(copy):
            try:
                interval_copy = copy()
            except Exception as exc:
                self._record_exception("Interval.copy() succeeds.", exc)
            else:
                self._check(True, "Interval.copy() succeeds.")
                copied_present = getattr(interval_copy, "present", None)
                copied_step = getattr(interval_copy, "step", None)
                copied_stop = getattr(interval_copy, "stop", None)
                self._check(
                    copied_present == getattr(interval, "present", None)
                    and copied_step == getattr(interval, "step", None)
                    and copied_stop == getattr(interval, "stop", None),
                    "Interval.copy() preserves present, step, and stop.",
                    "Make copy() return an interval snapshot with matching present, step, and stop.",
                )

                if callable(increment):
                    try:
                        before = interval_copy.present
                        interval_copy.increment(0.125)
                    except Exception as exc:
                        self._record_exception("Interval.increment(dt) succeeds on a copy.", exc)
                    else:
                        after = interval_copy.present
                        self._check(
                            after > before,
                            "Interval.increment(dt) advances present on a copy.",
                            "Make increment(dt) increase present by the accepted step size.",
                        )

    def _audit_tolerance(self, tolerance: Any) -> None:
        self._check(callable(getattr(tolerance, "bound", None)), "Tolerance provides bound(scale).", "Pass stark.Tolerance(...) or add bound(scale).")
        self._check(callable(getattr(tolerance, "ratio", None)), "Tolerance provides ratio(error, scale).", "Add ratio(error, scale) for adaptive schemes.")
        self._check(callable(getattr(tolerance, "accepts", None)), "Tolerance provides accepts(error, scale).", "Add accepts(error, scale) if you want compatibility with STARK's tolerance interface.")

    def _audit_scheme(self, scheme: Any) -> None:
        self._check(
            callable(scheme),
            "Scheme provides __call__(interval, state, tolerance).",
            "Add __call__(interval, state, tolerance) returning the accepted step size.",
        )
        self._check(
            callable(getattr(scheme, "snapshot_state", None)),
            "Scheme provides snapshot_state(state).",
            "Add snapshot_state(state) so snapshot integration can clone the state.",
        )
        self._check(
            callable(getattr(scheme, "set_apply_delta_safety", None)),
            "Scheme provides set_apply_delta_safety(enabled).",
            "Add set_apply_delta_safety(enabled) to control alias-safe state updates.",
        )

    def _audit_marcher(self, marcher: Any, *, snapshots: bool) -> None:
        self._check(callable(marcher), "Marcher object is callable.", "Provide a marcher(interval, state) callable.")
        if snapshots:
            self._check(
                callable(getattr(marcher, "snapshot_state", None)),
                "Marcher provides snapshot_state(state) for snapshot integration.",
                "Use Marcher(...) or add snapshot_state(state) before calling Integrator(...).",
            )

    def _exercise_derivative(self, derivative: Any, state: Any, translation: Any) -> None:
        try:
            derivative(state, translation)
        except Exception as exc:
            self._record_exception("Derivative(state, translation) can be called.", exc)
        else:
            self._check(True, "Derivative(state, translation) can be called.")

    def _exercise_copy_state(self, workbench: Any, dst: Any, src: Any) -> None:
        copy_state = getattr(workbench, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(dst, src)
        except Exception as exc:
            self._record_exception("Workbench.copy_state(dst, src) can copy a provided state.", exc)
        else:
            self._check(True, "Workbench.copy_state(dst, src) can copy a provided state.")

    def _exercise_workbench_copy(self, workbench: Any, dst: Any, src: Any) -> None:
        copy_state = getattr(workbench, "copy_state", None)
        if not callable(copy_state):
            return
        try:
            copy_state(dst, src)
        except Exception as exc:
            self._record_exception("Workbench.copy_state(dst, src) succeeds on blank states.", exc)
        else:
            self._check(True, "Workbench.copy_state(dst, src) succeeds on blank states.")

    def _exercise_translation_apply(self, translation: Any, state: Any, result: Any) -> None:
        if not callable(translation):
            return
        try:
            translation(state, result)
        except Exception as exc:
            self._record_exception("Translation(origin, result) can be called.", exc)
        else:
            self._check(True, "Translation(origin, result) can be called.")

        norm = getattr(translation, "norm", None)
        if callable(norm):
            try:
                value = norm()
            except Exception as exc:
                self._record_exception("Translation.norm() succeeds.", exc)
            else:
                self._check(
                    isinstance(value, Real),
                    "Translation.norm() returns a real number.",
                    "Return a float-like norm from Translation.norm().",
                )

    def _exercise_linear_combine(self, translation: Any, sample_translation: Any) -> None:
        linear_combine = getattr(translation, "linear_combine", None)
        if not isinstance(linear_combine, (list, tuple)):
            return

        for index, combine in enumerate(linear_combine, start=1):
            if not callable(combine):
                continue
            out = sample_translation
            args: list[Any] = [out]
            for term in range(index):
                args.extend([float(term + 1), sample_translation])
            arity_name = "scale" if index == 1 else f"combine{index}"
            try:
                combine(*args)
            except Exception as exc:
                self._record_exception(f"Translation {arity_name} can be called.", exc)
            else:
                self._check(True, f"Translation {arity_name} can be called.")

    @classmethod
    def require_scheme_inputs(cls, derivative: Any, workbench: Any, translation: Any) -> None:
        cls(derivative=derivative, workbench=workbench, translation=translation, exercise=False).raise_if_invalid()

    @classmethod
    def require_marcher_inputs(cls, scheme: Any, tolerance: Any, apply_delta_safety: Any) -> None:
        auditor = cls(scheme=scheme, tolerance=tolerance, snapshots=True, exercise=False)
        auditor._check(
            isinstance(apply_delta_safety, bool),
            "apply_delta_safety is a bool.",
            "Set apply_delta_safety to True or False.",
        )
        auditor.raise_if_invalid()

    @classmethod
    def require_integration_inputs(cls, marcher: Any, interval: Any, state: Any, *, snapshots: bool) -> None:
        del state
        cls(marcher=marcher, interval=interval, snapshots=snapshots, exercise=False).raise_if_invalid()

    def __str__(self) -> str:
        headers = ("Object", "Required behavior", "Present")
        checks = sorted(
            enumerate(self.checks),
            key=lambda indexed: (
                self._object_order(self._object_name(indexed[1][1])),
                indexed[0],
            ),
        )
        rows = [self._display_row(summary, passed) for _index, (passed, summary, _detail) in checks]
        widths = [
            max(len(row[index]) for row in (headers, *rows))
            for index in range(len(headers))
        ]

        def render(row: tuple[str, str, str]) -> str:
            return " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))

        lines = [
            "STARK audit checklist",
            render(headers),
            "-+-".join("-" * width for width in widths),
        ]
        for _index, (passed, summary, detail) in checks:
            lines.append(render(self._display_row(summary, passed)))
            if detail and not passed:
                lines.append(f"  detail: {detail}")

        status = "ready" if self.ok else "incomplete"
        lines.append(f"Overall: {status}.")
        return "\n".join(lines)

    @classmethod
    def _display_row(cls, summary: str, passed: bool) -> tuple[str, str, str]:
        return cls._object_name(summary), summary.rstrip("."), "yes" if passed else "no"

    @staticmethod
    def _object_name(summary: str) -> str:
        if summary.startswith("Derivative"):
            return "Derivative"
        if summary.startswith("Translation"):
            return "Translation"
        if summary.startswith("Workbench"):
            return "Workbench"
        if summary.startswith("Tolerance"):
            return "Tolerance"
        if summary.startswith("Interval"):
            return "Interval"
        if summary.startswith("Marcher"):
            return "Marcher"
        if summary.startswith("Scheme"):
            return "Scheme"
        if summary.startswith("apply_delta_safety"):
            return "Marcher"
        return "Interface"

    @staticmethod
    def _object_order(object_name: str) -> int:
        order = {
            "Interval": 0,
            "Derivative": 1,
            "Translation": 2,
            "Workbench": 3,
            "Scheme": 4,
            "Tolerance": 5,
            "Marcher": 6,
            "Interface": 7,
        }
        return order.get(object_name, order["Interface"])


__all__ = ["AuditError", "Auditor"]
