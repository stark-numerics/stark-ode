"""ASV benchmarks for catalogue-driven IVP runs."""

from __future__ import annotations

from benchmarks.builders import BenchmarkBuilder


BENCHMARK_IVP_BUILDER = BenchmarkBuilder()
BENCHMARK_IVP_SMOKE_NAMES = BENCHMARK_IVP_BUILDER.smoke_axis_names()
BENCHMARK_IVP_REPRESENTATIVE_NAMES = BENCHMARK_IVP_BUILDER.representative_axis_names()
BENCHMARK_IVP_FULL_NAMES = BENCHMARK_IVP_BUILDER.full_axis_names()


class BenchmarkTimeIVPSetupBase:
    """Time preparation of a benchmark IVP."""

    param_names = ["axis"]

    def benchmark_setup(self, axis: str) -> None:
        BENCHMARK_IVP_BUILDER.run(BENCHMARK_IVP_BUILDER.axis(axis)).ivp()


class BenchmarkTimeIVPFirstSolveBase:
    """Time one prepared IVP solve including first-use backend costs."""

    param_names = ["axis"]

    def setup(self, axis: str) -> None:
        self.ivp = BENCHMARK_IVP_BUILDER.run(BENCHMARK_IVP_BUILDER.axis(axis)).ivp()

    def benchmark_first_solve(self, axis: str) -> None:
        del axis
        self.ivp.final_result()


class BenchmarkTimeIVPRepeatSolveBase:
    """Time repeated solves after IVP preparation and one warmup solve."""

    param_names = ["axis"]

    def setup(self, axis: str) -> None:
        self.ivp = BENCHMARK_IVP_BUILDER.run(BENCHMARK_IVP_BUILDER.axis(axis)).ivp()
        self.ivp.final_result()

    def benchmark_repeat_solve(self, axis: str) -> None:
        del axis
        self.ivp.final_result()


class BenchmarkTimeIVPErrorBase:
    """Track final-state error for benchmark problems with references."""

    param_names = ["axis"]

    def setup(self, axis: str) -> None:
        self.run = BENCHMARK_IVP_BUILDER.run(BENCHMARK_IVP_BUILDER.axis(axis))

    def benchmark_error(self, axis: str) -> float:
        del axis
        result = self.run.ivp().final_result()
        error = self.run.problem.error(result.state)
        if error is None:
            return 0.0
        return error


class BenchmarkTimeIVPSmokeSetup(BenchmarkTimeIVPSetupBase):
    params = [BENCHMARK_IVP_SMOKE_NAMES]

    def time_setup(self, axis: str) -> None:
        self.benchmark_setup(axis)


class BenchmarkTimeIVPSmokeFirstSolve(BenchmarkTimeIVPFirstSolveBase):
    params = [BENCHMARK_IVP_SMOKE_NAMES]

    def time_first_solve(self, axis: str) -> None:
        self.benchmark_first_solve(axis)


class BenchmarkTimeIVPSmokeRepeatSolve(BenchmarkTimeIVPRepeatSolveBase):
    params = [BENCHMARK_IVP_SMOKE_NAMES]

    def time_repeat_solve(self, axis: str) -> None:
        self.benchmark_repeat_solve(axis)


class BenchmarkTimeIVPSmokeError(BenchmarkTimeIVPErrorBase):
    params = [BENCHMARK_IVP_SMOKE_NAMES]

    def track_error(self, axis: str) -> float:
        return self.benchmark_error(axis)


class BenchmarkTimeIVPRepresentativeSetup(BenchmarkTimeIVPSetupBase):
    params = [BENCHMARK_IVP_REPRESENTATIVE_NAMES]

    def time_setup(self, axis: str) -> None:
        self.benchmark_setup(axis)


class BenchmarkTimeIVPRepresentativeFirstSolve(BenchmarkTimeIVPFirstSolveBase):
    params = [BENCHMARK_IVP_REPRESENTATIVE_NAMES]

    def time_first_solve(self, axis: str) -> None:
        self.benchmark_first_solve(axis)


class BenchmarkTimeIVPRepresentativeRepeatSolve(BenchmarkTimeIVPRepeatSolveBase):
    params = [BENCHMARK_IVP_REPRESENTATIVE_NAMES]

    def time_repeat_solve(self, axis: str) -> None:
        self.benchmark_repeat_solve(axis)


class BenchmarkTimeIVPRepresentativeError(BenchmarkTimeIVPErrorBase):
    params = [BENCHMARK_IVP_REPRESENTATIVE_NAMES]

    def track_error(self, axis: str) -> float:
        return self.benchmark_error(axis)


class BenchmarkTimeIVPFullSetup(BenchmarkTimeIVPSetupBase):
    params = [BENCHMARK_IVP_FULL_NAMES]

    def time_setup(self, axis: str) -> None:
        self.benchmark_setup(axis)


class BenchmarkTimeIVPFullFirstSolve(BenchmarkTimeIVPFirstSolveBase):
    params = [BENCHMARK_IVP_FULL_NAMES]

    def time_first_solve(self, axis: str) -> None:
        self.benchmark_first_solve(axis)


class BenchmarkTimeIVPFullRepeatSolve(BenchmarkTimeIVPRepeatSolveBase):
    params = [BENCHMARK_IVP_FULL_NAMES]

    def time_repeat_solve(self, axis: str) -> None:
        self.benchmark_repeat_solve(axis)


class BenchmarkTimeIVPFullError(BenchmarkTimeIVPErrorBase):
    params = [BENCHMARK_IVP_FULL_NAMES]

    def track_error(self, axis: str) -> float:
        return self.benchmark_error(axis)
