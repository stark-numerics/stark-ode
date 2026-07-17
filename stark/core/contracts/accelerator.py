"""Protocols for acceleration backends.

Accelerators compile plain callable kernels. They do not inspect whether a
callable is a dynamics, linearizer, resolvent helper, or generated algebra
kernel; that role information belongs to the object constructing the callable. This
keeps the accelerator contract small enough for users and backend authors to
reason about.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Protocol, TypeVar, overload

from stark.core.contracts.contract_audit import AuditRecorder

AcceleratorTarget = TypeVar("AcceleratorTarget", bound=Callable[..., Any])


class Accelerator(Protocol):
    """Public protocol for STARK acceleration workers."""

    name: ClassVar[str]
    strict: bool

    @overload
    def compile(
        self,
        function: None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[[AcceleratorTarget], AcceleratorTarget]:
        ...

    @overload
    def compile(
        self,
        function: AcceleratorTarget,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget:
        ...

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> AcceleratorTarget | Callable[[AcceleratorTarget], AcceleratorTarget]:
        ...

    def compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        ...


class AcceleratorAudit:
    """Audit whether an accelerator satisfies STARK's public contract."""

    def __call__(self, recorder: AuditRecorder, accelerator: Any, *, exercise: bool = True) -> None:
        recorder.check(
            isinstance(getattr(accelerator, "name", None), str),
            "Accelerator provides a string name.",
            "Add accelerator.name so diagnostics can report the selected acceleration worker.",
        )
        recorder.check(
            isinstance(getattr(accelerator, "strict", None), bool),
            "Accelerator reports strict mode as a bool.",
            "Add accelerator.strict so STARK can distinguish best-effort from strict acceleration.",
        )

        compile_method = getattr(accelerator, "compile", None)
        recorder.check(
            callable(compile_method),
            "Accelerator provides compile(function=None, **options).",
            "Add compile(...) so support kernels can be compiled or traced during setup.",
        )

        compile_examples = getattr(accelerator, "compile_examples", None)
        recorder.check(
            callable(compile_examples),
            "Accelerator provides compile_examples(function, *examples).",
            "Add compile_examples(...) so representative examples can be compiled during setup.",
        )

        if not exercise or not callable(compile_method):
            return

        def probe(value: float) -> float:
            return value

        try:
            compiled_probe = compile_method(probe, label="audit")
        except Exception as exc:
            recorder.record_exception(
                "Accelerator.compile(function) succeeds.",
                exc,
                "Ensure compile(function) returns a callable worker or the original function.",
            )
            return
        else:
            recorder.check(
                callable(compiled_probe),
                "Accelerator.compile(function) returns a callable.",
                "Return a callable from compile(function).",
            )

        if not callable(compile_examples):
            return

        try:
            compiled = compile_examples(compiled_probe, (1.0,))
        except Exception as exc:
            recorder.record_exception(
                "Accelerator.compile_examples(function, *examples) succeeds.",
                exc,
                "Ensure compile_examples(...) accepts representative example arguments.",
            )
        else:
            recorder.check(
                callable(compiled),
                "Accelerator.compile_examples(function, *examples) returns a callable.",
                "Return the compiled or original callable from compile_examples(...).",
            )


__all__ = [
    "AcceleratorAudit",
    "Accelerator",
    "AcceleratorTarget",
]
