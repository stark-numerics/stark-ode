"""Structured comparison tooling for STARK experiments."""

from stark.comparison.comparator import Comparator
from stark.comparison.models import (
    ComparatorEntry,
    ComparatorProblem,
    ComparatorReport,
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
    "ComparatorEntry",
    "ComparatorProblem",
    "ComparatorReport",
    "Comparison",
    "ComparisonBreakdown",
    "ComparisonDiagnostics",
    "ComparisonHotspot",
    "ComparisonProfile",
    "ComparisonResult",
    "ComparisonTiming",
]


