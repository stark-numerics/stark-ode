"""Structured comparison reports for STARK solver experiments.

The comparison package owns reusable report data models and the development
runner that compares several configured steppers on one problem. It is the
programmatic counterpart to the longer narrative examples under
`competition/`.
"""

from stark.comparison.runner import ComparisonRunner
from stark.comparison.models import (
    Comparison,
    ComparisonBreakdown,
    ComparisonDiagnostics,
    ComparisonEntry,
    ComparisonEntryLike,
    ComparisonEntryStepper,
    ComparisonHotspot,
    ComparisonProblem,
    ComparisonProblemLike,
    ComparisonProblemManual,
    ComparisonProfile,
    ComparisonReport,
    ComparisonResult,
    ComparisonTiming,
    ObservedStepperBuilder,
    StepperBuilder,
)

__all__ = [
    "ComparisonRunner",
    "Comparison",
    "ComparisonBreakdown",
    "ComparisonDiagnostics",
    "ComparisonEntry",
    "ComparisonEntryLike",
    "ComparisonEntryStepper",
    "ComparisonHotspot",
    "ComparisonProblem",
    "ComparisonProblemLike",
    "ComparisonProblemManual",
    "ComparisonProfile",
    "ComparisonReport",
    "ComparisonResult",
    "ComparisonTiming",
    "ObservedStepperBuilder",
    "StepperBuilder",
]
