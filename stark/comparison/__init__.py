"""Structured comparison reports for STARK solver experiments.

The comparison package owns reusable report data models and the development
runner that compares several configured marchers on one problem. It is the
programmatic counterpart to the longer narrative examples under
`examples/comparison/`.
"""

from stark.comparison.runner import ComparisonRunner
from stark.comparison.models import (
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
