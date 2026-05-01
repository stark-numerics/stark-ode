from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeVar

from stark.contracts.audit_support import AuditRecorder


CompiledCallable = TypeVar("CompiledCallable", bound=Callable[..., Any])


class AccelerationRole(str, Enum):
    """Typed acceleration request roles used by STARK workers."""

    DERIVATIVE = "derivative"
    LINEARIZER = "linearizer"
    SUPPORT = "support"


@dataclass(frozen=True, slots=True)
class AccelerationRequest:
    """Describe one acceleration-resolution request."""

    role: AccelerationRole
    label: str | None = None
    values: dict[str, Any] = field(default_factory=dict, repr=False)


class SupportsAcceleration(Protocol):
    """Optional hook for user objects that can return accelerated variants."""

    def accelerated(self, accelerator: "AcceleratorLike", request: AccelerationRequest) -> Any:
        ...


class AccelerationBackend(Protocol):
    """Minimal backend protocol used by built-in accelerator workers."""

    name: str

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        ...

    def compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        ...


class AcceleratorLike(Protocol):
    """Public protocol for STARK acceleration workers."""

    name: str
    strict: bool

    def decorate(self, function: CompiledCallable | None = None, /, **kwargs: Any) -> Callable[..., Any]:
        ...

    def compile_examples(self, function: CompiledCallable, *signatures: Any) -> CompiledCallable:
        ...

    def resolve(self, target: Any, request: AccelerationRequest) -> Any:
        ...

    def resolve_derivative(self, derivative: Any) -> Any:
        ...

    def resolve_linearizer(self, linearizer: Any) -> Any:
        ...

    def resolve_support(self, worker: Any, *, label: str | None = None, **values: Any) -> Any:
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

        decorate = getattr(accelerator, "decorate", None)
        recorder.check(
            callable(decorate),
            "Accelerator provides decorate(function=None, **kwargs).",
            "Add decorate(...) so support kernels can be compiled or traced during setup.",
        )

        compile_examples = getattr(accelerator, "compile_examples", None)
        recorder.check(
            callable(compile_examples),
            "Accelerator provides compile_examples(function, *signatures).",
            "Add compile_examples(...) so representative signatures can be compiled during setup.",
        )
        resolve = getattr(accelerator, "resolve", None)
        recorder.check(
            callable(resolve),
            "Accelerator provides resolve(target, request).",
            "Add resolve(target, request) so STARK can ask for accelerated worker variants explicitly.",
        )

        if not exercise or not callable(decorate):
            return

        def probe(value: float) -> float:
            return value

        try:
            decorated = decorate(probe)
        except Exception as exc:
            recorder.record_exception(
                "Accelerator.decorate(function) succeeds.",
                exc,
                "Ensure decorate(function) returns a callable worker or the original function.",
            )
            return
        else:
            recorder.check(
                callable(decorated),
                "Accelerator.decorate(function) returns a callable.",
                "Return a callable from decorate(function).",
            )

        if not callable(compile_examples):
            return

        try:
            compiled = compile_examples(decorated, (1.0,))
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

        if not callable(resolve):
            return

        request = AccelerationRequest(AccelerationRole.SUPPORT, label="audit")
        try:
            resolved = resolve(probe, request)
        except Exception as exc:
            recorder.record_exception(
                "Accelerator.resolve(target, request) succeeds.",
                exc,
                "Ensure resolve(target, request) returns an accelerated variant or the original target.",
            )
        else:
            recorder.check(
                callable(resolved),
                "Accelerator.resolve(target, request) returns a callable target.",
                "Return the accelerated variant or original callable from resolve(target, request).",
            )


__all__ = [
    "AccelerationBackend",
    "AccelerationRequest",
    "AccelerationRole",
    "AcceleratorAudit",
    "AcceleratorLike",
    "CompiledCallable",
    "SupportsAcceleration",
]
