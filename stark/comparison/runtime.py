from __future__ import annotations

import cProfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from stark.comparison.models import (
    ComparatorEntry,
    ComparatorProblem,
    ComparisonBreakdown,
    ComparisonDiagnostics,
    ComparisonHotspot,
    ComparisonProfile,
    ComparisonTiming,
    ProfileCategory,
)
from stark.integrate import Integrator


@dataclass(slots=True)
class BakedRun:
    state: Any
    checkpoints: list[Any]
    steps: int
    elapsed: float


@dataclass(slots=True)
class BakedEntry:
    state: Any
    checkpoints: list[Any]
    steps: int
    timing: ComparisonTiming
    diagnostics: ComparisonDiagnostics
    profile: ComparisonProfile


class CountingMarcher:
    __slots__ = ("marcher", "steps")

    def __init__(self, marcher: Any) -> None:
        self.marcher = marcher
        self.steps = 0

    def __call__(self, interval: Any, state: Any) -> None:
        self.steps += 1
        self.marcher(interval, state)

    def snapshot_state(self, state: Any) -> Any:
        snapshot_state = getattr(self.marcher, "snapshot_state", None)
        if not callable(snapshot_state):
            raise TypeError(
                "Comparator checkpoint comparison requires marcher.snapshot_state(state). "
                "Use Marcher(...) or add snapshot_state(state) to the custom marcher."
            )
        return snapshot_state(state)

    def set_safety(self, safety: Any) -> None:
        if hasattr(self.marcher, "set_safety"):
            self.marcher.set_safety(safety)

    def set_apply_delta_safety(self, enabled: bool) -> None:
        if hasattr(self.marcher, "set_apply_delta_safety"):
            self.marcher.set_apply_delta_safety(enabled)


class ProfileSurvey:
    __slots__ = ()

    def __call__(
        self,
        profiler: cProfile.Profile,
        marcher: Any,
        profile_category: ProfileCategory | None = None,
    ) -> ComparisonProfile:
        return ComparisonProfile(
            breakdown=self._breakdown(profiler, profile_category),
            note=self._note(marcher),
            custom_hotspots=self._custom_hotspots(profiler, marcher),
        )

    def _breakdown(
        self,
        profiler: cProfile.Profile,
        profile_category: ProfileCategory | None,
    ) -> ComparisonBreakdown:
        problem = method = resolvent = inverter = framework = other = 0.0
        for filename, lineno, function_name, _total_calls, self_time, _cumulative_time in _profile_stats(profiler):
            if function_name == "disable":
                continue
            category = profile_category(filename, lineno, function_name) if profile_category is not None else None
            if category is None:
                category = self._category(filename)
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
    def _category(filename: str) -> str:
        if filename.startswith("~"):
            return "other"
        path = filename.replace("/", "\\").lower()
        if "\\stark\\schemes\\" in path:
            return "method"
        if "\\stark\\resolvents\\" in path:
            return "resolvent"
        if "\\stark\\inverters\\" in path or "\\stark\\linear_algebra\\" in path or "\\stark\\block\\" in path:
            return "inverter"
        if "\\stark\\" in path:
            return "framework"
        return "problem"

    @staticmethod
    def _note(marcher: Any) -> str | None:
        scheme = getattr(marcher, "scheme", None)
        if scheme is None:
            return (
                "Custom entry: the profile buckets are approximate and cannot reliably "
                "separate method logic from resolvent and problem work."
            )
        module_name = type(scheme).__module__
        if module_name.startswith("stark.schemes."):
            return None
        return (
            f"Custom scheme {type(scheme).__name__}: the profile buckets are approximate and "
            "cannot reliably separate method logic from resolvent and problem work."
        )

    def _custom_hotspots(self, profiler: cProfile.Profile, marcher: Any) -> list[ComparisonHotspot]:
        if self._note(marcher) is None:
            return []

        hotspots: list[ComparisonHotspot] = []
        for filename, lineno, function_name, total_calls, self_time, cumulative_time in _profile_stats(profiler):
            if function_name == "disable":
                continue
            path = filename.replace("/", "\\").lower()
            if "\\stark\\" in path:
                continue
            if filename == "~" and function_name.startswith("<method 'disable'"):
                continue
            hotspots.append(
                ComparisonHotspot(
                    location=self._hotspot_location(filename, lineno, function_name),
                    self_time=self_time,
                    cumulative_time=cumulative_time,
                    calls=total_calls,
                )
            )

        hotspots.sort(key=lambda hotspot: (hotspot.cumulative_time, hotspot.self_time), reverse=True)
        return hotspots[:5]

    @staticmethod
    def _hotspot_location(filename: str, lineno: int, function_name: str) -> str:
        if filename == "~":
            return function_name
        return f"{Path(filename).name}:{lineno} {function_name}"


class Comparer:
    __slots__ = ("announce", "problem", "profile_survey", "repeats")

    def __init__(self, problem: ComparatorProblem, repeats: int, announce: Any | None = None) -> None:
        self.problem = problem
        self.repeats = repeats
        self.announce = announce
        self.profile_survey = ProfileSurvey()

    def __call__(self, entry: ComparatorEntry) -> BakedEntry:
        self._announce(f"Comparing {entry.name}...")

        started = perf_counter()
        marcher = CountingMarcher(entry.make_marcher())
        integrator = entry.build_integrator() if entry.build_integrator is not None else Integrator()
        setup_elapsed = perf_counter() - started

        warmup = self._run_once(marcher, integrator)
        self._announce(f"Warmup {entry.name}: steps={warmup.steps}, elapsed={warmup.elapsed:.3f}s")

        durations = []
        for repeat in range(self.repeats):
            timed = self._run_once(marcher, integrator)
            durations.append(timed.elapsed)
            self._announce(
                f"Timed {entry.name} repeat {repeat + 1}/{self.repeats}: "
                f"steps={timed.steps}, elapsed={timed.elapsed:.3f}s"
            )

        profile = self._profile_once(marcher, integrator, entry.profile_category)
        diagnostics = self.problem.diagnostics(warmup.state) if self.problem.diagnostics is not None else None

        return BakedEntry(
            state=warmup.state,
            checkpoints=warmup.checkpoints,
            steps=warmup.steps,
            timing=ComparisonTiming(
                setup=setup_elapsed,
                warmup=warmup.elapsed,
                median=float(median(durations)),
                minimum=float(min(durations)),
            ),
            diagnostics=ComparisonDiagnostics.coerce(diagnostics),
            profile=profile,
        )

    def _run_once(self, marcher: CountingMarcher, integrator: Integrator) -> BakedRun:
        state = self.problem.build_state()
        interval = self.problem.build_interval()
        marcher.steps = 0
        checkpoints: list[Any] = []
        started = perf_counter()
        for _interval, _state in integrator.live(marcher, interval, state, checkpoints=self.problem.checkpoints):
            if self.problem.checkpoints is not None:
                checkpoints.append(marcher.snapshot_state(state))
        elapsed = perf_counter() - started
        return BakedRun(state=state, checkpoints=checkpoints, steps=marcher.steps, elapsed=elapsed)

    def _profile_once(
        self,
        marcher: CountingMarcher,
        integrator: Integrator,
        profile_category: ProfileCategory | None,
    ) -> ComparisonProfile:
        state = self.problem.build_state()
        interval = self.problem.build_interval()
        marcher.steps = 0
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            for _interval, _state in integrator.live(marcher, interval, state, checkpoints=self.problem.checkpoints):
                pass
        finally:
            profiler.disable()

        profile = self.profile_survey(profiler, marcher.marcher, profile_category)
        breakdown = profile.breakdown
        if breakdown.profiled > 0.0:
            self._announce(
                "Profiled self-time breakdown: "
                f"problem={100.0 * breakdown.problem / breakdown.profiled:.1f}%, "
                f"method={100.0 * breakdown.method / breakdown.profiled:.1f}%, "
                f"framework={100.0 * breakdown.framework / breakdown.profiled:.1f}%"
            )
        return profile

    def _announce(self, message: str) -> None:
        if self.announce is not None:
            self.announce(message)


def _profile_stats(profiler: cProfile.Profile):
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


__all__ = ["BakedEntry", "BakedRun", "Comparer", "CountingMarcher", "ProfileSurvey"]



