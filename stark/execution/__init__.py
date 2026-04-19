"""Runtime execution policy and acceleration workers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "AdaptiveController",
    "Executor",
    "Regulator",
    "Safety",
    "SchemeTolerance",
    "Tolerance",
    "current_executor",
]


def __getattr__(name: str):
    if name == "AdaptiveController":
        module = import_module("stark.execution.adaptive_controller")
        return getattr(module, name)
    if name in {"Executor", "current_executor"}:
        module = import_module("stark.execution.executor")
        return getattr(module, name)
    if name == "Regulator":
        module = import_module("stark.execution.regulator")
        return getattr(module, name)
    if name == "Safety":
        module = import_module("stark.execution.safety")
        return getattr(module, name)
    if name in {"Tolerance", "SchemeTolerance"}:
        module = import_module("stark.execution.tolerance")
        return getattr(module, name)
    raise AttributeError(name)


