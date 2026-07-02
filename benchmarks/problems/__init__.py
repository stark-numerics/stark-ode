"""Reusable benchmark problem definitions."""

from benchmarks.problems.harmonic_oscillator import BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR
from benchmarks.problems.large_linear_chain import BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN
from benchmarks.problems.problem import BenchmarkProblemDefinition
from benchmarks.problems.reaction_diffusion_array import BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY
from benchmarks.problems.robertson import BENCHMARK_PROBLEM_ROBERTSON
from benchmarks.problems.scalar_decay import BENCHMARK_PROBLEM_SCALAR_DECAY
from benchmarks.problems.van_der_pol_stiff import BENCHMARK_PROBLEM_VAN_DER_POL_STIFF


__all__ = [
    "BENCHMARK_PROBLEM_HARMONIC_OSCILLATOR",
    "BENCHMARK_PROBLEM_LARGE_LINEAR_CHAIN",
    "BENCHMARK_PROBLEM_REACTION_DIFFUSION_ARRAY",
    "BENCHMARK_PROBLEM_ROBERTSON",
    "BENCHMARK_PROBLEM_SCALAR_DECAY",
    "BENCHMARK_PROBLEM_VAN_DER_POL_STIFF",
    "BenchmarkProblemDefinition",
]

