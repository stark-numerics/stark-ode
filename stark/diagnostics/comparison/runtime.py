"""Runtime machinery behind `ComparisonRunner`.

This module keeps the operational parts of comparison separate from the public
report models: step counting, checkpoint capture, repeated timing, optional
monitor observation, and cProfile bucketing.
"""

from __future__ import annotations

import cProfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from stark.diagnostics.comparison.models import (
    ComparisonEntry,
    ComparisonProblem,
    ComparisonBreakdown,
    ComparisonDiagnostics,
    ComparisonHotspot,
    ComparisonProfile,
    ComparisonTiming,
    ProfileCategory,
)
from stark.core.contracts import IntervalLike, State
from stark.core.integrator.integrator import Integrator
from stark.core.integrator.stepper import IntegratorStepper
from stark.diagnostics.monitor import Monitor, MonitorSummary


@dataclass(slots=True)
class ComparisonRunRecord:
    """Raw output from one comparison solve."""

    state: Any
    checkpoints: list[Any]
    steps: int
    elapsed: float


@dataclass(slots=True)
class ComparisonEntryEvaluation:
    """Prepared evidence for one entry after observation, timing, and profiling."""

    state: Any
    checkpoints: list[Any]
    steps: int
    timing: ComparisonTiming
    diagnostics: ComparisonDiagnostics
    profile: ComparisonProfile
    monitor_summary: MonitorSummary | None


class ComparisonStepperCounting:
    """Wrap a stepper so comparison runs can count accepted step calls."""

    __slots__ = ("stepper", "steps")

    def __init__(self, stepper: Any) -> None:
        self.stepper = stepper
        self.steps = 0

    def __call__(self, interval: IntervalLike, state: State) -> None:
        self.steps += 1
        self.stepper(interval, state)

    def snapshot_state(self, state: State) -> State:
        snapshot_state = getattr(self.stepper, "snapshot_state", None)
        if not callable(snapshot_state):
            raise TypeError(
                "ComparisonRunner checkpoint comparison requires stepper.snapshot_state(state). "
                "Use IntegratorStepper(...) or add snapshot_state(state) to the custom stepper."
            )
        return cast(State, snapshot_state(state))


class ComparisonProfileSurvey:
    """Convert cProfile statistics into comparison profile buckets."""

    __slots__ = ()

    def __call__(
        self,
        profiler: cProfile.Profile,
        stepper: Any,
        profile_category: ProfileCategory | None = None,
    ) -> ComparisonProfile:
        return ComparisonProfile(
            breakdown=self.breakdown(profiler, profile_category),
            note=self.note(stepper),
            custom_hotspots=self.custom_hotspots(profiler, stepper),
        )

    def breakdown(
        self,
        profiler: cProfile.Profile,
        profile_category: ProfileCategory | None,
    ) -> ComparisonBreakdown:
        problem = method = resolvent = inverter = framework = other = 0.0
        for filename, lineno, function_name, _total_calls, self_time, _cumulative_time in self.profile_stats(profiler):
            if function_name == "disable":
                continue
            category = profile_category(filename, lineno, function_name) if profile_category is not None else None
            if category is None:
                category = self.category(filename)
            if category == "problem":
                problem += self_time
            elif category in {"method", "scheme"}:
                method += self_time
            elif category in {"resolvent", "resolver"}:
                resolvent += self_time
            elif category == "inverter":
                inverter += self_time
            elif category == "framework":
                framework += self_time
            else:
                other += self_time
        total = problem + method + resolvent + inverter + framework + other
        return ComparisonBreakdown(
            profiled=total,
            problem=problem,
            method=method,
            resolvent=resolvent,
            inverter=inverter,
            framework=framework,
            other=other,
        )

    @staticmethod
    def category(filename: str) -> str:
        if filename.startswith("~"):
            return "other"
        path = filename.replace("/", "\\").lower()
        if "\\stark\\methods\\schemes\\" in path:
            return "method"
        if "\\stark\\methods\\resolvents\\" in path:
            return "resolvent"
        if (
            "\\stark\\methods\\inverters\\" in path
            or "\\stark\\core\\block\\" in path
        ):
            return "inverter"
        if "\\stark\\" in path:
            return "framework"
        return "problem"

    @staticmethod
    def note(stepper: Any) -> str | None:
        scheme = getattr(stepper, "scheme", None)
        if scheme is None:
            return (
                "Custom entry: the profile buckets are approximate and cannot reliably "
                "separate method logic from resolvent and problem work."
            )
        module_name = type(scheme).__module__
        if module_name.startswith("stark.methods.schemes."):
            return None
        return (
            f"Custom scheme {type(scheme).__name__}: the profile buckets are approximate and "
            "cannot reliably separate method logic from resolvent and problem work."
        )

    def custom_hotspots(self, profiler: cProfile.Profile, stepper: Any) -> list[ComparisonHotspot]:
        if self.note(stepper) is None:
            return []

        hotspots: list[ComparisonHotspot] = []
        for filename, lineno, function_name, total_calls, self_time, cumulative_time in self.profile_stats(profiler):
            if function_name == "disable":
                continue
            path = filename.replace("/", "\\").lower()
            if "\\stark\\" in path:
                continue
            if filename == "~" and function_name.startswith("<method 'disable'"):
                continue
            hotspots.append(
                ComparisonHotspot(
                    location=self.hotspot_location(filename, lineno, function_name),
                    self_time=self_time,
                    cumulative_time=cumulative_time,
                    calls=total_calls,
                )
            )

        hotspots.sort(key=lambda hotspot: (hotspot.cumulative_time, hotspot.self_time), reverse=True)
        return hotspots[:5]

    @staticmethod
    def hotspot_location(filename: str, lineno: int, function_name: str) -> str:
        if filename == "~":
            return function_name
        return f"{Path(filename).name}:{lineno} {function_name}"

    @staticmethod
    def profile_stats(profiler: cProfile.Profile):
        for entry in profiler.getstats():
            code = entry.code
            if isinstance(code, str):
                filename = "~"
                lineno = 0
                function_name = code
            else:
                filename = getattr(code, "co_filename", "~")
                lineno = getattr(code, "co_firstlineno", 0)
                function_name = getattr(code, "co_name", str(code))
            yield (
                filename,
                lineno,
                function_name,
                entry.reccallcount + entry.callcount,
                entry.inlinetime,
                entry.totaltime,
            )


class ComparisonEntryEvaluator:
    """Prepare, observe, time, and profile one comparison entry."""

    __slots__ = ("announce", "problem", "profile_survey", "repeats")

    def __init__(self, problem: ComparisonProblem, repeats: int, announce: Any | None = None) -> None:
        self.problem = problem
        self.repeats = repeats
        self.announce = announce
        self.profile_survey = ComparisonProfileSurvey()

    def __call__(self, entry: ComparisonEntry) -> ComparisonEntryEvaluation:
        self.announce_message(f"Comparing {entry.name}...")

        started = perf_counter()
        stepper = ComparisonStepperCounting(entry.make_stepper(self.problem.ivp))
        integrator = entry.make_integrator(self.problem.ivp)
        if integrator is None:
            integrator = Integrator()
        setup_elapsed = perf_counter() - started

        observed_state, observed_checkpoints, observed_steps, monitor_summary = self.observe_once(entry, integrator)
        self.announce_message(f"Observed {entry.name}: steps={observed_steps}")

        warmup = self.run_once(stepper, integrator)
        self.announce_message(f"Warmup {entry.name}: steps={warmup.steps}, elapsed={warmup.elapsed:.3f}s")

        durations = []
        for repeat in range(self.repeats):
            timed = self.run_once(stepper, integrator)
            durations.append(timed.elapsed)
            self.announce_message(
                f"Timed {entry.name} repeat {repeat + 1}/{self.repeats}: "
                f"steps={timed.steps}, elapsed={timed.elapsed:.3f}s"
            )

        profile = self.profile_once(stepper, integrator, entry.profile_category)
        diagnostics = self.problem.diagnostics(observed_state) if self.problem.diagnostics is not None else None

        return ComparisonEntryEvaluation(
            state=observed_state,
            checkpoints=observed_checkpoints,
            steps=observed_steps,
            timing=ComparisonTiming(
                setup=setup_elapsed,
                warmup=warmup.elapsed,
                median=float(median(durations)),
                minimum=float(min(durations)),
            ),
            diagnostics=ComparisonDiagnostics.coerce(diagnostics),
            profile=profile,
            monitor_summary=monitor_summary,
        )

    def observe_once(
        self,
        entry: ComparisonEntry,
        integrator: Integrator,
    ) -> tuple[Any, list[Any], int, MonitorSummary | None]:
        monitor = Monitor()
        stepper = ComparisonStepperCounting(entry.make_observed_stepper(monitor, self.problem.ivp))
        observed = self.run_once(stepper, integrator)
        if entry.build_observed_stepper is None:
            return observed.state, observed.checkpoints, observed.steps, None

        summary = monitor.summary()
        if (
            summary.scheme.step_count == 0
            and summary.resolvent.solve_count == 0
            and summary.inverter.solve_count == 0
        ):
            return observed.state, observed.checkpoints, observed.steps, None
        return observed.state, observed.checkpoints, observed.steps, summary

    def run_once(self, stepper: ComparisonStepperCounting, integrator: Integrator) -> ComparisonRunRecord:
        state = self.problem.build_state()
        interval = self.problem.build_interval()
        stepper.steps = 0
        checkpoints: list[Any] = []
        started = perf_counter()
        integrator_stepper = cast(IntegratorStepper, stepper)
        for _interval, _state in integrator.mutating_trajectory(integrator_stepper, interval, state, checkpoints=self.problem.checkpoints):
            if self.problem.checkpoints is not None:
                checkpoints.append(stepper.snapshot_state(state))
        elapsed = perf_counter() - started
        return ComparisonRunRecord(state=state, checkpoints=checkpoints, steps=stepper.steps, elapsed=elapsed)

    def profile_once(
        self,
        stepper: ComparisonStepperCounting,
        integrator: Integrator,
        profile_category: ProfileCategory | None,
    ) -> ComparisonProfile:
        state = self.problem.build_state()
        interval = self.problem.build_interval()
        stepper.steps = 0
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            integrator_stepper = cast(IntegratorStepper, stepper)
            for _interval, _state in integrator.mutating_trajectory(integrator_stepper, interval, state, checkpoints=self.problem.checkpoints):
                pass
        finally:
            profiler.disable()

        profile = self.profile_survey(profiler, stepper.stepper, profile_category)
        breakdown = profile.breakdown
        if breakdown.profiled > 0.0:
            self.announce_message(
                "Profiled self-time breakdown: "
                f"problem={100.0 * breakdown.problem / breakdown.profiled:.1f}%, "
                f"method={100.0 * breakdown.method / breakdown.profiled:.1f}%, "
                f"framework={100.0 * breakdown.framework / breakdown.profiled:.1f}%"
            )
        return profile

    def announce_message(self, message: str) -> None:
        if self.announce is not None:
            self.announce(message)

__all__ = [
    "ComparisonEntryEvaluation",
    "ComparisonEntryEvaluator",
    "ComparisonRunRecord",
    "ComparisonStepperCounting",
    "ComparisonProfileSurvey",
]



