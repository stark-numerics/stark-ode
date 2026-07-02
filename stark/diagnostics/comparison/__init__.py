"""Structured comparison reports for STARK solver runs.

The comparison package owns reusable report data models and the runner that
compares several configured `Method` choices on one `SystemIVP` problem. It is
for tight method-selection runs before a longer scientific solve.
"""

from stark.diagnostics.comparison.runner import ComparisonRunner
from stark.diagnostics.comparison.models import (
    Comparison,
    ComparisonBreakdown,
    ComparisonDiagnostics,
    ComparisonEntry,
    ComparisonHotspot,
    ComparisonProblem,
    ComparisonProfile,
    ComparisonReport,
    ComparisonResult,
    ComparisonTiming,
)

__all__ = [
    "ComparisonRunner",
    "Comparison",
    "ComparisonBreakdown",
    "ComparisonDiagnostics",
    "ComparisonEntry",
    "ComparisonHotspot",
    "ComparisonProblem",
    "ComparisonProfile",
    "ComparisonReport",
    "ComparisonResult",
    "ComparisonTiming",
]
