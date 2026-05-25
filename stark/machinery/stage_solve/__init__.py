"""Workers for staged scheme stepping and stage-bound workspaces."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ResolventCoupledCollocationStep",
    "ImExStepper",
    "SchemeWorkspace",
    "SequentialDIRKResolventStep",
    "ShiftedOneStageResolventStep",
]


def __getattr__(name: str):
    if name == "SchemeWorkspace":
        module = import_module("stark.machinery.stage_solve.workspace")
        return getattr(module, name)
    if name in {
        "ResolventCoupledCollocationStep",
        "ImExStepper",
        "SequentialDIRKResolventStep",
        "ShiftedOneStageResolventStep",
    }:
        module = import_module("stark.machinery.stage_solve.workers")
        return getattr(module, name)
    raise AttributeError(name)


