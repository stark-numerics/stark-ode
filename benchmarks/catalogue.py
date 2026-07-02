"""Benchmark-facing catalogue of representative problems and engines.

The package method catalogue answers "which method stacks exist?". This module
answers "what should we throw them at?". It deliberately keeps problem and
engine metadata separate so ASV can later build a cartesian benchmark grid with
clear compatibility filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from benchmarks.problems import (
    BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR,
    BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN,
    BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY,
    BENCHMARK_PROBLEM_ROBERTSON,
    BENCHMARK_PROBLEM_SCALAR_DECAY,
    BENCHMARK_PROBLEM_VAN_DER_POL_STIFF,
    BenchmarkProblemDefinition,
)
from stark.methods.catalogue import (
    METHOD_CATALOGUE,
    MethodCatalogue,
    MethodCatalogueBenchmarkTier,
    MethodCatalogueProblemClass,
    MethodCatalogueStack,
)


class BenchmarkCatalogueTier(str, Enum):
    """How often a benchmark entry should be included."""

    SMOKE = "smoke"
    REPRESENTATIVE = "representative"
    EXHAUSTIVE = "exhaustive"


class BenchmarkCatalogueProblemFeature(str, Enum):
    """Problem properties that matter for benchmark coverage."""

    ARRAY_STATE = "array-state"
    COUPLED_FIELDS = "coupled-fields"
    LARGE_STATE = "large-state"
    NONSTIFF = "nonstiff"
    NUMPY_STATE = "numpy-state"
    OSCILLATORY = "oscillatory"
    SPLIT = "split"
    STIFF = "stiff"


class BenchmarkCatalogueReference(str, Enum):
    """How a benchmark problem can check accuracy."""

    EXACT = "exact"
    HIGH_PRECISION = "high-precision"
    INVARIANT = "invariant"
    NONE = "none"


class BenchmarkCatalogueScale(str, Enum):
    """Approximate state size for benchmark planning."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class BenchmarkCatalogueEngineFamily(str, Enum):
    """Engine families that benchmarks may compare."""

    NATIVE = "native"
    NUMPY = "numpy"
    JAX = "jax"
    CUPY = "cupy"


class BenchmarkCatalogueNamed(Protocol):
    """Protocol for catalogue entries that can be looked up by name."""

    name: str


@dataclass(frozen=True, slots=True)
class BenchmarkCatalogueProblem:
    """Representative problem recipe metadata.

    The builder is intentionally not stored here yet. The catalogue first names
    the benchmark coverage shape; ASV-specific builders can then be added
    without changing the public meaning of the entry.
    """

    name: str
    summary: str
    problem_classes: tuple[MethodCatalogueProblemClass, ...]
    features: tuple[BenchmarkCatalogueProblemFeature, ...]
    scale: BenchmarkCatalogueScale
    reference: BenchmarkCatalogueReference
    benchmark_tier: BenchmarkCatalogueTier
    definition: BenchmarkProblemDefinition

    def accepts_method_stack(self, stack: MethodCatalogueStack) -> bool:
        """Return whether a method stack is worth trying on this problem."""

        return (
            stack.problem_class == MethodCatalogueProblemClass.GENERAL
            or stack.problem_class in self.problem_classes
        )

    def accepts_engine(self, engine: BenchmarkCatalogueEngine) -> bool:
        """Return whether an engine belongs in this problem's benchmark grid."""

        if (
            BenchmarkCatalogueProblemFeature.NUMPY_STATE in self.features
            and engine.family
            not in {
                BenchmarkCatalogueEngineFamily.NATIVE,
                BenchmarkCatalogueEngineFamily.NUMPY,
            }
        ):
            return False
        is_array_problem = (
            BenchmarkCatalogueProblemFeature.ARRAY_STATE in self.features
            or BenchmarkCatalogueProblemFeature.LARGE_STATE in self.features
        )
        if (
            BenchmarkCatalogueProblemFeature.NUMPY_STATE in self.features
            and engine.family == BenchmarkCatalogueEngineFamily.NATIVE
        ):
            return False
        return not is_array_problem or engine.prefers_array_state


@dataclass(frozen=True, slots=True)
class BenchmarkCatalogueEngine:
    """Benchmark metadata for one engine configuration."""

    name: str
    family: BenchmarkCatalogueEngineFamily
    summary: str
    benchmark_tier: BenchmarkCatalogueTier
    optional_dependency: str | None = None
    accelerator: str | None = None
    prefers_array_state: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkCatalogueAxis:
    """One benchmarkable problem/method/engine combination."""

    problem: BenchmarkCatalogueProblem
    method_stack: MethodCatalogueStack
    engine: BenchmarkCatalogueEngine


@dataclass(frozen=True, slots=True)
class BenchmarkCatalogue:
    """Catalogue used to build benchmark matrices."""

    problems: tuple[BenchmarkCatalogueProblem, ...]
    engines: tuple[BenchmarkCatalogueEngine, ...]

    def problem(self, name: str) -> BenchmarkCatalogueProblem:
        return self.entry_by_name(self.problems, name)

    def engine(self, name: str) -> BenchmarkCatalogueEngine:
        return self.entry_by_name(self.engines, name)

    def compatible_method_stacks(
        self,
        problem: str | BenchmarkCatalogueProblem,
        method_catalogue: MethodCatalogue = METHOD_CATALOGUE,
    ) -> tuple[MethodCatalogueStack, ...]:
        problem_entry = self.problem(problem) if isinstance(problem, str) else problem
        return tuple(
            stack
            for stack in method_catalogue.stacks
            if stack.benchmark_tier != MethodCatalogueBenchmarkTier.OMIT
            and problem_entry.accepts_method_stack(stack)
        )

    def axes(
        self,
        method_catalogue: MethodCatalogue = METHOD_CATALOGUE,
    ) -> tuple[BenchmarkCatalogueAxis, ...]:
        return tuple(
            BenchmarkCatalogueAxis(problem, method_stack, engine)
            for problem in self.problems
            for method_stack in self.compatible_method_stacks(problem, method_catalogue)
            for engine in self.engines
            if problem.accepts_engine(engine)
        )

    @staticmethod
    def entry_by_name[T: BenchmarkCatalogueNamed](entries: tuple[T, ...], name: str) -> T:
        for entry in entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


BENCHMARK_CATALOGUE_PROBLEMS = (
    BenchmarkCatalogueProblem(
        name="scalar-decay",
        summary="Tiny non-stiff scalar IVP with an exact exponential reference.",
        problem_classes=(MethodCatalogueProblemClass.GENERAL, MethodCatalogueProblemClass.NONSTIFF),
        features=(
            BenchmarkCatalogueProblemFeature.NONSTIFF,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
        ),
        scale=BenchmarkCatalogueScale.TINY,
        reference=BenchmarkCatalogueReference.EXACT,
        benchmark_tier=BenchmarkCatalogueTier.SMOKE,
        definition=BENCHMARK_PROBLEM_SCALAR_DECAY,
    ),
    BenchmarkCatalogueProblem(
        name="harmonic-oscillator",
        summary="Non-stiff coupled oscillator that checks phase and invariant drift.",
        problem_classes=(MethodCatalogueProblemClass.GENERAL, MethodCatalogueProblemClass.NONSTIFF),
        features=(
            BenchmarkCatalogueProblemFeature.COUPLED_FIELDS,
            BenchmarkCatalogueProblemFeature.NONSTIFF,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
            BenchmarkCatalogueProblemFeature.OSCILLATORY,
        ),
        scale=BenchmarkCatalogueScale.TINY,
        reference=BenchmarkCatalogueReference.INVARIANT,
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        definition=BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR,
    ),
    BenchmarkCatalogueProblem(
        name="robertson",
        summary="Small stiff chemical kinetics problem with a high-precision reference.",
        problem_classes=(MethodCatalogueProblemClass.STIFF,),
        features=(
            BenchmarkCatalogueProblemFeature.COUPLED_FIELDS,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
            BenchmarkCatalogueProblemFeature.STIFF,
        ),
        scale=BenchmarkCatalogueScale.TINY,
        reference=BenchmarkCatalogueReference.HIGH_PRECISION,
        benchmark_tier=BenchmarkCatalogueTier.SMOKE,
        definition=BENCHMARK_PROBLEM_ROBERTSON,
    ),
    BenchmarkCatalogueProblem(
        name="van-der-pol-stiff",
        summary="Stiff coupled oscillator for implicit and adaptive-step behaviour.",
        problem_classes=(MethodCatalogueProblemClass.STIFF,),
        features=(
            BenchmarkCatalogueProblemFeature.COUPLED_FIELDS,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
            BenchmarkCatalogueProblemFeature.OSCILLATORY,
            BenchmarkCatalogueProblemFeature.STIFF,
        ),
        scale=BenchmarkCatalogueScale.TINY,
        reference=BenchmarkCatalogueReference.HIGH_PRECISION,
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        definition=BENCHMARK_PROBLEM_VAN_DER_POL_STIFF,
    ),
    BenchmarkCatalogueProblem(
        name="reaction-diffusion-array",
        summary="Array-valued split problem for IMEX schemes and array-backed engines.",
        problem_classes=(MethodCatalogueProblemClass.SPLIT,),
        features=(
            BenchmarkCatalogueProblemFeature.ARRAY_STATE,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
            BenchmarkCatalogueProblemFeature.SPLIT,
            BenchmarkCatalogueProblemFeature.STIFF,
        ),
        scale=BenchmarkCatalogueScale.MEDIUM,
        reference=BenchmarkCatalogueReference.HIGH_PRECISION,
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        definition=BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY,
    ),
    BenchmarkCatalogueProblem(
        name="large-linear-chain",
        summary="Large non-stiff array problem for backend throughput comparisons.",
        problem_classes=(MethodCatalogueProblemClass.GENERAL, MethodCatalogueProblemClass.NONSTIFF),
        features=(
            BenchmarkCatalogueProblemFeature.ARRAY_STATE,
            BenchmarkCatalogueProblemFeature.LARGE_STATE,
            BenchmarkCatalogueProblemFeature.NONSTIFF,
            BenchmarkCatalogueProblemFeature.NUMPY_STATE,
        ),
        scale=BenchmarkCatalogueScale.LARGE,
        reference=BenchmarkCatalogueReference.EXACT,
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        definition=BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN,
    ),
)


BENCHMARK_CATALOGUE_ENGINES = (
    BenchmarkCatalogueEngine(
        name="native",
        family=BenchmarkCatalogueEngineFamily.NATIVE,
        summary="Pure Python native scalar engine.",
        benchmark_tier=BenchmarkCatalogueTier.SMOKE,
    ),
    BenchmarkCatalogueEngine(
        name="numpy",
        family=BenchmarkCatalogueEngineFamily.NUMPY,
        summary="NumPy engine without accelerator.",
        benchmark_tier=BenchmarkCatalogueTier.SMOKE,
        prefers_array_state=True,
    ),
    BenchmarkCatalogueEngine(
        name="numpy-numba",
        family=BenchmarkCatalogueEngineFamily.NUMPY,
        summary="NumPy engine with generated Numba acceleration.",
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        optional_dependency="numba",
        accelerator="numba",
        prefers_array_state=True,
    ),
    BenchmarkCatalogueEngine(
        name="jax",
        family=BenchmarkCatalogueEngineFamily.JAX,
        summary="JAX engine using generated Algebraist paths.",
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        optional_dependency="jax",
        prefers_array_state=True,
    ),
    BenchmarkCatalogueEngine(
        name="cupy",
        family=BenchmarkCatalogueEngineFamily.CUPY,
        summary="CuPy engine using generated GPU-oriented Algebraist paths.",
        benchmark_tier=BenchmarkCatalogueTier.REPRESENTATIVE,
        optional_dependency="cupy",
        prefers_array_state=True,
    ),
)


BENCHMARK_CATALOGUE = BenchmarkCatalogue(
    problems=BENCHMARK_CATALOGUE_PROBLEMS,
    engines=BENCHMARK_CATALOGUE_ENGINES,
)


__all__ = [
    "BENCHMARK_CATALOGUE",
    "BENCHMARK_CATALOGUE_ENGINES",
    "BENCHMARK_CATALOGUE_PROBLEMS",
    "BenchmarkCatalogue",
    "BenchmarkCatalogueAxis",
    "BenchmarkCatalogueEngine",
    "BenchmarkCatalogueEngineFamily",
    "BenchmarkCatalogueProblem",
    "BenchmarkCatalogueProblemFeature",
    "BenchmarkCatalogueReference",
    "BenchmarkCatalogueScale",
    "BenchmarkCatalogueTier",
]
