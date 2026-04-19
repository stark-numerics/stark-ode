from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator

from stark.accelerators import AcceleratorAbsent
from stark.contracts.acceleration import AcceleratorLike
from stark.execution.adaptive_controller import AdaptiveController
from stark.execution.regulator import Regulator
from stark.execution.safety import Safety
from stark.execution.tolerance import SchemeTolerance, Tolerance


_CURRENT_EXECUTOR: ContextVar["Executor | None"] = ContextVar("stark_executor", default=None)


@dataclass(slots=True)
class Executor:
    """
    Runtime execution worker for STARK.

    The executor carries cross-cutting runtime policy such as tolerance, safety,
    adaptive-step regulation, and acceleration backend selection.
    """

    tolerance: Tolerance = field(default_factory=SchemeTolerance)
    safety: Safety = field(default_factory=Safety)
    regulator: Regulator = field(default_factory=Regulator)
    accelerator: AcceleratorLike = field(default_factory=AcceleratorAbsent)
    values: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        from stark.auditor import Auditor

        Auditor(accelerator=self.accelerator, exercise=False).raise_if_invalid()

    def __repr__(self) -> str:
        return (
            "Executor("
            f"tolerance={self.tolerance!r}, "
            f"safety={self.safety!r}, "
            f"regulator={self.regulator!r}, "
            f"accelerator={self.accelerator!r}, "
            f"values={self.values!r})"
        )

    def __str__(self) -> str:
        extras = "" if not self.values else f", extras={tuple(sorted(self.values))!r}"
        return f"{self.tolerance}, {self.safety}, {self.regulator}, accelerator={self.accelerator}{extras}"

    def __getattr__(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:  # pragma: no cover - ordinary attribute fallback
            raise AttributeError(name) from exc

    def bound(self, scale: float) -> float:
        return self.tolerance.bound(scale)

    def ratio(self, error: float, scale: float) -> float:
        return self.tolerance.ratio(error, scale)

    def accepts(self, error: float, scale: float) -> bool:
        return self.tolerance.accepts(error, scale)

    def adaptive_controller(self, fallback: Regulator | None = None) -> AdaptiveController:
        regulator = self.regulator if self.regulator is not None else (fallback if fallback is not None else Regulator())
        return AdaptiveController(regulator)

    def with_updates(self, **updates: Any) -> "Executor":
        values = dict(self.values)
        tolerance = self.tolerance
        safety = self.safety
        regulator = self.regulator
        accelerator = self.accelerator

        if "tolerance" in updates:
            tolerance = updates.pop("tolerance")
        if "safety" in updates:
            safety = updates.pop("safety")
        if "regulator" in updates:
            regulator = updates.pop("regulator")
        if "accelerator" in updates:
            accelerator = updates.pop("accelerator")

        values.update(updates)
        return Executor(
            tolerance=tolerance,
            safety=safety,
            regulator=regulator,
            accelerator=accelerator,
            values=values,
        )

    @contextmanager
    def use(self) -> Iterator["Executor"]:
        token: Token[Executor | None] = _CURRENT_EXECUTOR.set(self)
        try:
            yield self
        finally:
            _CURRENT_EXECUTOR.reset(token)

    @classmethod
    def current(cls) -> "Executor | None":
        return _CURRENT_EXECUTOR.get()

    @classmethod
    def resolve(
        cls,
        executor: "Executor | Tolerance | None" = None,
        *,
        tolerance: Tolerance | None = None,
        safety: Safety | None = None,
        regulator: Regulator | None = None,
        accelerator: AcceleratorLike | None = None,
        **values: Any,
    ) -> "Executor":
        base = executor if isinstance(executor, cls) else cls.current()
        if base is None:
            base = cls()

        if executor is not None and not isinstance(executor, cls):
            tolerance = executor

        updates: dict[str, Any] = {}
        if tolerance is not None:
            updates["tolerance"] = tolerance
        if safety is not None:
            updates["safety"] = safety
        if regulator is not None:
            updates["regulator"] = regulator
        if accelerator is not None:
            updates["accelerator"] = accelerator
        updates.update(values)
        return base if not updates else base.with_updates(**updates)


def current_executor() -> Executor | None:
    return Executor.current()


__all__ = ["Executor", "current_executor"]









