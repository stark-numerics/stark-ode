from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from stark.comparison.models import (
    ComparatorReport,
    Comparison,
    ComparisonDiagnostics,
    ComparisonHotspot,
    ComparisonProfile,
    ComparisonResult,
)


class ComparisonTableWriter:
    __slots__ = ()

    def __call__(self, headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
        rows = list(rows)
        widths = [len(header) for header in headers]
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], len(value))

        lines = [
            " | ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)),
            "-+-".join("-" * width for width in widths),
        ]
        for row in rows:
            lines.append(" | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)))
        return "\n".join(lines)


class ComparisonDiagnosticsWriter:
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table = ComparisonTableWriter()

    def __call__(self, diagnostics: ComparisonDiagnostics) -> str:
        if not diagnostics:
            return "no diagnostics"
        return self.table(("diagnostic", "value"), [(name, self._format(value)) for name, value in diagnostics])

    @staticmethod
    def _format(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)


class ComparisonWriter:
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table = ComparisonTableWriter()

    def __call__(self, comparison: Comparison) -> str:
        headers = ("entry", *comparison.labels)
        rows = []
        for row_name, row_values in comparison.rows():
            rows.append((row_name, *[f"{value:.6e}" for value in row_values]))
        lines = [self.table(headers, rows)]
        if comparison.note:
            lines.extend(["", comparison.note])
        return "\n".join(lines)


class ComparisonProfileWriter:
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table = ComparisonTableWriter()

    def __call__(self, profile: ComparisonProfile) -> str:
        breakdown = profile.breakdown
        lines = [
            self.table(
                ("bucket", "time", "share"),
                [
                    ("problem", f"{breakdown.problem:.6f}s", self._format_percent(breakdown.problem, breakdown.profiled)),
                    ("method", f"{breakdown.method:.6f}s", self._format_percent(breakdown.method, breakdown.profiled)),
                    ("resolvent", f"{breakdown.resolvent:.6f}s", self._format_percent(breakdown.resolvent, breakdown.profiled)),
                    ("inverter", f"{breakdown.inverter:.6f}s", self._format_percent(breakdown.inverter, breakdown.profiled)),
                    ("framework", f"{breakdown.framework:.6f}s", self._format_percent(breakdown.framework, breakdown.profiled)),
                    ("other", f"{breakdown.other:.6f}s", self._format_percent(breakdown.other, breakdown.profiled)),
                ],
            )
        ]
        if profile.note:
            lines.extend(["", profile.note])
        if profile.custom_hotspots:
            lines.extend(
                [
                    "",
                    self.table(
                        ("location", "self", "cumulative", "calls"),
                        [
                            (
                                hotspot.location,
                                f"{hotspot.self_time:.6f}s",
                                f"{hotspot.cumulative_time:.6f}s",
                                str(hotspot.calls),
                            )
                            for hotspot in profile.custom_hotspots
                        ],
                    ),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _format_percent(value: float, total: float) -> str:
        if total <= 0.0:
            return "0.0%"
        return f"{100.0 * value / total:.1f}%"


class ComparisonResultWriter:
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table = ComparisonTableWriter()

    def __call__(self, result: ComparisonResult) -> str:
        lines = [f"{result.name}", f"steps={result.steps}", str(result.timing)]
        if result.metadata:
            lines.extend(
                [
                    "",
                    "Configuration",
                    self.table(("field", "value"), [(name, str(value)) for name, value in result.metadata.items()]),
                ]
            )
        if result.diagnostics:
            lines.extend(["", "Diagnostics", str(result.diagnostics)])
        lines.extend(["", "Profile", str(result.profile)])
        return "\n".join(lines)


class ComparatorReportWriter:
    __slots__ = ("table",)

    def __init__(self) -> None:
        self.table = ComparisonTableWriter()

    def __call__(self, report: ComparatorReport) -> str:
        lines = [f"{report.problem_name} comparison", ""]
        if report.description:
            lines.extend([report.description, ""])

        lines.extend(
            [
                f"Each entry is warmed once, timed {report.repeats} times, and profiled once.",
                "Setup excludes one-time cross-entry builder warmup costs." if report.prewarmed_builders else "Setup includes builder costs directly.",
                "",
                "Timing Table",
                self.table(
                    ("entry", "setup", "warmup", "median", "min"),
                    [
                        (
                            result.name,
                            f"{result.timing.setup:.6f}s",
                            f"{result.timing.warmup:.6f}s",
                            f"{result.timing.median:.6f}s",
                            f"{result.timing.minimum:.6f}s",
                        )
                        for result in report.results
                    ],
                ),
                "",
                "Pairwise final-state differences",
                self._render_matrix(report.final_differences),
            ]
        )

        metadata_names = self._metadata_names(report)
        if metadata_names:
            lines.extend(
                [
                    "",
                    "Configuration Table",
                    "Rows show entry metadata supplied through ComparatorEntry.metadata.",
                    self._render_metadata_matrix(report, metadata_names),
                ]
            )

        if self._diagnostic_names(report):
            lines.extend(
                [
                    "",
                    "Diagnostics Table",
                    "Rows show problem-supplied final-state diagnostics from ComparatorProblem.diagnostics(...).",
                    self._render_diagnostics_matrix(report),
                ]
            )

        if report.trajectory_differences is not None:
            lines.extend(["", "Pairwise trajectory differences"])
            lines.append(self._render_matrix(report.trajectory_differences))

        lines.extend(
            [
                "",
                "Profile breakdown by self time",
                "Rows show the share of profiled self time attributed to problem kernels,",
                "method logic, nonlinear resolvent work, linear inverter work,",
                "general STARK framework overhead, and uncategorized remainder.",
                self._render_breakdown_matrix(report.results),
            ]
        )

        notes = [result.profile.note for result in report.results if result.profile.note]
        if notes:
            lines.extend(["", "Profile notes"])
            seen = set()
            for result in report.results:
                if result.profile.note and result.profile.note not in seen:
                    lines.append(f"- {result.profile.note}")
                    seen.add(result.profile.note)

        hotspot_results = [result for result in report.results if result.profile.custom_hotspots]
        if hotspot_results:
            lines.extend(
                [
                    "",
                    "Custom entry hotspots",
                    "Rows show the top non-STARK profile entries for custom marchers or runtime",
                    "configurations, ordered by cumulative time so users can see where their own",
                    "code is spending time.",
                ]
            )
            for result in hotspot_results:
                lines.extend(["", result.name, self._render_hotspot_table(result.profile.custom_hotspots)])

        return "\n".join(lines)

    def _render_matrix(self, comparison: Comparison) -> str:
        headers = ("entry", *comparison.labels)
        rows = []
        for row_name, row_values in comparison.rows():
            rows.append((row_name, *[f"{value:.6e}" for value in row_values]))
        lines = [self.table(headers, rows)]
        if comparison.note:
            lines.extend(["", comparison.note])
        return "\n".join(lines)

    def _render_metadata_matrix(self, report: ComparatorReport, metadata_names: list[str]) -> str:
        headers = ("entry", *metadata_names)
        rows = []
        for result in report.results:
            rows.append((result.name, *[self._format_diagnostic(result.metadata.get(name, "")) for name in metadata_names]))
        return self.table(headers, rows)

    def _render_diagnostics_matrix(self, report: ComparatorReport) -> str:
        names = self._diagnostic_names(report)
        headers = ("entry", "steps", *names)
        rows = []
        for result in report.results:
            rows.append((result.name, str(result.steps), *[self._format_diagnostic(result.diagnostics.get(name, "")) for name in names]))
        return self.table(headers, rows)

    def _render_breakdown_matrix(self, results: list[ComparisonResult]) -> str:
        rows = []
        for result in results:
            breakdown = result.profile.breakdown
            rows.append(
                (
                    result.name,
                    f"{breakdown.profiled:.6f}s",
                    self._format_percent(breakdown.problem, breakdown.profiled),
                    self._format_percent(breakdown.method, breakdown.profiled),
                    self._format_percent(breakdown.resolvent, breakdown.profiled),
                    self._format_percent(breakdown.inverter, breakdown.profiled),
                    self._format_percent(breakdown.framework, breakdown.profiled),
                    self._format_percent(breakdown.other, breakdown.profiled),
                )
            )
        return self.table(("entry", "profiled", "problem", "method", "resolvent", "inverter", "framework", "other"), rows)

    def _render_hotspot_table(self, hotspots: list[ComparisonHotspot]) -> str:
        return self.table(
            ("location", "self", "cumulative", "calls"),
            [
                (hotspot.location, f"{hotspot.self_time:.6f}s", f"{hotspot.cumulative_time:.6f}s", str(hotspot.calls))
                for hotspot in hotspots
            ],
        )

    @staticmethod
    def _diagnostic_names(report: ComparatorReport) -> list[str]:
        names: list[str] = []
        for result in report.results:
            for name in result.diagnostics.names():
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _metadata_names(report: ComparatorReport) -> list[str]:
        names: list[str] = []
        for result in report.results:
            for name in result.metadata:
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _format_diagnostic(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _format_percent(value: float, total: float) -> str:
        if total <= 0.0:
            return "0.0%"
        return f"{100.0 * value / total:.1f}%"


__all__ = [
    "ComparatorReportWriter",
    "ComparisonDiagnosticsWriter",
    "ComparisonProfileWriter",
    "ComparisonResultWriter",
    "ComparisonTableWriter",
    "ComparisonWriter",
]


