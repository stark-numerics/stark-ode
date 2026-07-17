"""Focused ASV benchmarks for engine-generated kernel consumption."""

from __future__ import annotations

from importlib.util import find_spec

from benchmarks.builders import BenchmarkBuilder


BENCHMARK_ENGINE_GENERATED_BUILDER = BenchmarkBuilder()
BENCHMARK_ENGINE_GENERATED_NUMPY_AXIS_NAMES = (
    "large-linear-chain/rk4/numpy",
    "large-linear-chain/rkck/numpy",
    "reaction-diffusion-array/imexeuler-newton-dense/numpy",
)
BENCHMARK_ENGINE_GENERATED_NUMBA_AXIS_NAMES = (
    "large-linear-chain/rk4/numpy-numba",
    "large-linear-chain/rkck/numpy-numba",
    "reaction-diffusion-array/imexeuler-newton-dense/numpy-numba",
)
BENCHMARK_ENGINE_GENERATED_AXIS_NAMES = (
    BENCHMARK_ENGINE_GENERATED_NUMPY_AXIS_NAMES
    + BENCHMARK_ENGINE_GENERATED_NUMBA_AXIS_NAMES
    if find_spec("numba") is not None
    else BENCHMARK_ENGINE_GENERATED_NUMPY_AXIS_NAMES
)


class BenchmarkTimeEngineGeneratedPrepare:
    """Time IVP preparation through engine algebra and scheme specialization."""

    param_names = ["axis"]
    params = [BENCHMARK_ENGINE_GENERATED_AXIS_NAMES]

    def time_prepare(self, axis: str) -> None:
        BENCHMARK_ENGINE_GENERATED_BUILDER.run(
            BENCHMARK_ENGINE_GENERATED_BUILDER.axis(axis)
        ).ivp()


class BenchmarkTimeEngineGeneratedFirstSolve:
    """Time a first solve after engine and scheme preparation."""

    param_names = ["axis"]
    params = [BENCHMARK_ENGINE_GENERATED_AXIS_NAMES]

    def setup(self, axis: str) -> None:
        self.ivp = BENCHMARK_ENGINE_GENERATED_BUILDER.run(
            BENCHMARK_ENGINE_GENERATED_BUILDER.axis(axis)
        ).ivp()

    def time_first_solve(self, axis: str) -> None:
        self.ivp.final_result()


class BenchmarkTimeEngineGeneratedRepeatSolve:
    """Time repeated solves after generated-kernel first-use costs."""

    param_names = ["axis"]
    params = [BENCHMARK_ENGINE_GENERATED_AXIS_NAMES]

    def setup(self, axis: str) -> None:
        self.ivp = BENCHMARK_ENGINE_GENERATED_BUILDER.run(
            BENCHMARK_ENGINE_GENERATED_BUILDER.axis(axis)
        ).ivp()
        self.ivp.final_result()

    def time_repeat_solve(self, axis: str) -> None:
        self.ivp.final_result()
