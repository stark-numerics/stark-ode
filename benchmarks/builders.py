"""Builders that turn benchmark catalogue entries into runnable IVPs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from benchmarks.catalogue import (
    BENCHMARK_CATALOGUE,
    BenchmarkCatalogue,
    BenchmarkCatalogueAxis,
    BenchmarkCatalogueEngine,
    BenchmarkCatalogueEngineFamily,
    BenchmarkCatalogueTier,
)
from benchmarks.problems import BenchmarkProblemDefinition
from stark import Method
from stark.core.configuration import Configuration
from stark.engines.shared.accelerators import AcceleratorNumba
from stark.methods import METHOD_CATALOGUE
from stark.methods.catalogue import MethodCatalogue, MethodCatalogueStack
from stark.problem.system.system import EngineFactory, SystemIVP


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """Runnable benchmark stack built from one catalogue axis."""

    axis: BenchmarkCatalogueAxis
    method: Method
    engine: EngineFactory
    problem: BenchmarkProblemDefinition

    def ivp(self, configuration: Configuration | None = None) -> SystemIVP:
        """Build a prepared IVP for this benchmark stack."""

        return self.problem.ivp(
            method=self.method,
            engine=self.engine,
            configuration=configuration,
        )


class BenchmarkBuilder:
    """Build benchmark runs from the problem, method, and engine catalogues."""

    def __init__(
        self,
        benchmark_catalogue: BenchmarkCatalogue = BENCHMARK_CATALOGUE,
        method_catalogue: MethodCatalogue = METHOD_CATALOGUE,
    ) -> None:
        self.benchmark_catalogue = benchmark_catalogue
        self.method_catalogue = method_catalogue

    def engine_factory(self, engine: str | BenchmarkCatalogueEngine) -> EngineFactory:
        """Return the engine factory for a benchmark engine entry."""

        engine_entry = (
            self.benchmark_catalogue.engine(engine)
            if isinstance(engine, str)
            else engine
        )
        if engine_entry.family == BenchmarkCatalogueEngineFamily.NATIVE:
            return self.import_engine("EngineNative")
        if engine_entry.family == BenchmarkCatalogueEngineFamily.NUMPY:
            engine_type = self.import_engine("EngineNumpy")
            if engine_entry.accelerator is None:
                return engine_type
            if engine_entry.accelerator != "numba":
                raise ValueError(f"Unknown NumPy benchmark accelerator: {engine_entry.accelerator!r}")

            def engine_numpy_with_accelerator(frame):
                return engine_type(frame, accelerator=AcceleratorNumba())

            return engine_numpy_with_accelerator
        if engine_entry.family == BenchmarkCatalogueEngineFamily.JAX:
            return self.import_engine("EngineJax")
        if engine_entry.family == BenchmarkCatalogueEngineFamily.CUPY:
            return self.import_engine("EngineCupy")
        raise ValueError(f"Unknown benchmark engine family: {engine_entry.family!r}")

    def run(self, axis: BenchmarkCatalogueAxis) -> BenchmarkRun:
        """Build a runnable benchmark from one catalogue axis."""

        return BenchmarkRun(
            axis=axis,
            method=self.method_catalogue.method(axis.method_stack),
            engine=self.engine_factory(axis.engine),
            problem=axis.problem.definition,
        )

    def axes(self) -> tuple[BenchmarkCatalogueAxis, ...]:
        """Return all currently compatible benchmark axes."""

        return self.benchmark_catalogue.axes(self.method_catalogue)

    def axes_for_tier(self, tier: BenchmarkCatalogueTier) -> tuple[BenchmarkCatalogueAxis, ...]:
        """Return axes whose problem and engine are no broader than `tier`."""

        allowed = self.allowed_tiers(tier)
        return tuple(
            axis
            for axis in self.axes()
            if axis.problem.benchmark_tier in allowed
            and axis.engine.benchmark_tier in allowed
            and axis.method_stack.benchmark_tier in allowed
        )

    def smoke_axes(self) -> tuple[BenchmarkCatalogueAxis, ...]:
        """Return a cheap beta-release benchmark subset."""

        return self.axes_for_tier(BenchmarkCatalogueTier.SMOKE)

    def smoke_axis_names(self) -> tuple[str, ...]:
        """Return stable names for the cheap beta-release benchmark subset."""

        return tuple(self.axis_name(axis) for axis in self.smoke_axes())

    def representative_axes(self) -> tuple[BenchmarkCatalogueAxis, ...]:
        """Return the representative benchmark subset."""

        return self.axes_for_tier(BenchmarkCatalogueTier.REPRESENTATIVE)

    def representative_axis_names(self) -> tuple[str, ...]:
        """Return stable names for the representative benchmark subset."""

        return tuple(self.axis_name(axis) for axis in self.representative_axes())

    def full_axes(self) -> tuple[BenchmarkCatalogueAxis, ...]:
        """Return the full benchmark matrix for currently compatible entries."""

        return self.axes_for_tier(BenchmarkCatalogueTier.EXHAUSTIVE)

    def full_axis_names(self) -> tuple[str, ...]:
        """Return stable names for the full benchmark matrix."""

        return tuple(self.axis_name(axis) for axis in self.full_axes())

    def axis(self, name: str) -> BenchmarkCatalogueAxis:
        """Return a benchmark axis by its stable parameter name."""

        for axis in self.axes():
            if self.axis_name(axis) == name:
                return axis
        raise KeyError(name)

    @staticmethod
    def axis_name(axis: BenchmarkCatalogueAxis) -> str:
        """Return the stable ASV parameter name for one benchmark axis."""

        return f"{axis.problem.name}/{axis.method_stack.name}/{axis.engine.name}"

    @staticmethod
    def allowed_tiers(tier: BenchmarkCatalogueTier) -> tuple[BenchmarkCatalogueTier, ...]:
        if tier == BenchmarkCatalogueTier.SMOKE:
            return (BenchmarkCatalogueTier.SMOKE,)
        if tier == BenchmarkCatalogueTier.REPRESENTATIVE:
            return (
                BenchmarkCatalogueTier.SMOKE,
                BenchmarkCatalogueTier.REPRESENTATIVE,
            )
        return (
            BenchmarkCatalogueTier.SMOKE,
            BenchmarkCatalogueTier.REPRESENTATIVE,
            BenchmarkCatalogueTier.EXHAUSTIVE,
        )

    @staticmethod
    def import_engine(name: str) -> Any:
        engines = import_module("stark.engines")
        try:
            return getattr(engines, name)
        except AttributeError as exc:
            raise ModuleNotFoundError(f"{name} is not available in this environment.") from exc


__all__ = [
    "BenchmarkBuilder",
    "BenchmarkRun",
]
