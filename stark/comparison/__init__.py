"""Structured comparison tooling for STARK experiments."""

from stark.comparison.comparator import Comparator
from stark.comparison.models import (
    ComparisonEntry,
    ComparisonProblem,
    ComparisonReport,
    Comparison,
    ComparisonBreakdown,
    ComparisonDiagnostics,
    ComparisonHotspot,
    ComparisonProfile,
    ComparisonResult,
    ComparisonTiming,
)

__all__ = [
    "Comparator",
    "ComparisonEntry",
    "ComparisonProblem",
    "ComparisonReport",
    "Comparison",
    "ComparisonBreakdown",
    "ComparisonDiagnostics",
    "ComparisonHotspot",
    "ComparisonProfile",
    "ComparisonResult",
    "ComparisonTiming",
]


