"""Executor-owned runtime policy and acceleration workers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "Executor",
    "ExecutorAdaptivity",
    "ExecutorSafety",
    "ExecutorTolerance",
]


def __getattr__(name: str):
    if name == "ExecutorAdaptivity":
        module = import_module("stark.executor.adaptivity")
        return getattr(module, name)
    if name == "Executor":
        module = import_module("stark.executor.executor")
        return getattr(module, name)
    if name == "ExecutorSafety":
        module = import_module("stark.executor.safety")
        return getattr(module, name)
    if name == "ExecutorTolerance":
        module = import_module("stark.executor.tolerance")
        return getattr(module, name)
    raise AttributeError(name)


