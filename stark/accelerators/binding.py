from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stark.contracts.acceleration import AcceleratorLike


@dataclass(slots=True)
class BoundDerivative:
    """Bind a raw derivative to an accelerator outside the hot path."""

    raw: Any
    resolved: Any = field(init=False)

    def __post_init__(self) -> None:
        self.resolved = self.raw

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.resolved = accelerator.resolve_derivative(self.raw)

    def __call__(self, interval, state, out) -> Any:
        return self.resolved(interval, state, out)


@dataclass(slots=True)
class BoundLinearizer:
    """Bind a raw linearizer to an accelerator outside the hot path."""

    raw: Any
    resolved: Any = field(init=False)

    def __post_init__(self) -> None:
        self.resolved = self.raw

    def bind_accelerator(self, accelerator: AcceleratorLike) -> None:
        self.resolved = accelerator.resolve_linearizer(self.raw)

    def __call__(self, interval, state, out) -> Any:
        return self.resolved(interval, state, out)


def bind_worker_tree(worker: Any, accelerator: AcceleratorLike, _seen: set[int] | None = None) -> None:
    """Recursively bind built-in STARK worker graphs to one accelerator."""

    if _seen is None:
        _seen = set()

    identifier = id(worker)
    if identifier in _seen:
        return
    _seen.add(identifier)

    binder = getattr(worker, "bind_accelerator", None)
    if callable(binder):
        binder(accelerator)

    if worker is None or isinstance(worker, str | bytes | bytearray | int | float | complex | bool):
        return
    if isinstance(worker, dict):
        for value in worker.values():
            bind_worker_tree(value, accelerator, _seen)
        return
    if isinstance(worker, list | tuple | set | frozenset):
        for item in worker:
            bind_worker_tree(item, accelerator, _seen)
        return

    module_name = type(worker).__module__
    if not module_name.startswith("stark."):
        return

    for name in _iter_slot_names(type(worker)):
        if name == "__weakref__":
            continue
        try:
            value = getattr(worker, name)
        except AttributeError:
            continue
        bind_worker_tree(value, accelerator, _seen)

    values = getattr(worker, "__dict__", None)
    if values is not None:
        for value in values.values():
            bind_worker_tree(value, accelerator, _seen)


def _iter_slot_names(cls: type[Any]) -> tuple[str, ...]:
    names: list[str] = []
    for base in cls.__mro__:
        slots = getattr(base, "__slots__", ())
        if isinstance(slots, str):
            names.append(slots)
            continue
        names.extend(slots)
    return tuple(dict.fromkeys(names))


__all__ = ["BoundDerivative", "BoundLinearizer", "bind_worker_tree"]
