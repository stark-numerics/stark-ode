from __future__ import annotations

"""Protocols for acceleration backends.

Accelerators compile plain callable kernels. They do not inspect whether a
callable is a derivative, linearizer, resolvent helper, or algebraist kernel;
that role information belongs to the object constructing the callable. This
keeps the accelerator contract small enough for users and backend authors to
reason about.
"""

from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from stark.contracts.audit_support import AuditRecorder


AcceleratorTarget = TypeVar("AcceleratorTarget", bound=Callable[..., Any])


class AccelerationBackend(Protocol):
    """Minimal backend protocol used by built-in accelerator workers."""

    name: str

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
        ...

    def compile_examples(self, function: AcceleratorTarget, *examples: Any) -> AcceleratorTarget:
        ...


class AcceleratorLike(Protocol):
    """Public protocol for STARK acceleration workers."""

    name: str
    strict: bool

    def compile(
        self,
        function: AcceleratorTarget | None = None,
        /,
        *,
        label: str | None = None,
        cache: bool | None = None,
        **options: Any,
    ) -> Callable[..., Any]:
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
            "Accelerator provides compile_examples(function, *signatures).",
            "Add compile_examples(...) so representative signatures can be compiled during setup.",
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
                "Accelerator.compile_examples(function, *signatures) succeeds.",
                exc,
                "Ensure compile_examples(...) accepts representative example signatures.",
            )
        else:
            recorder.check(
                callable(compiled),
                "Accelerator.compile_examples(function, *signatures) returns a callable.",
                "Return the compiled or original callable from compile_examples(...).",
            )


__all__ = [
    "AccelerationBackend",
    "AcceleratorAudit",
    "AcceleratorLike",
    "AcceleratorTarget",
]
