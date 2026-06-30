"""Configuration protocol and defaults consumed by scheme implementations."""

from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass, field
from stark.core.contracts.scheme_predictor import SchemePredictorLike
from stark.core.tolerance import Tolerance
from stark.methods.schemes.predictor import SchemePredictorKnown

@dataclass(frozen=False, slots=True)
class SchemeConfiguration(Protocol):
    """Scheme configuration shape required by fixed and adaptive schemes."""

    scheme_tolerance: Tolerance
    adaptive_scheme_safety: float
    adaptive_scheme_min_factor: float
    adaptive_scheme_max_factor: float
    adaptive_scheme_error_exponent: float
    adaptive_scheme_maximum_rejections: int | None
    scheme_predictor: SchemePredictorLike | None

@dataclass(frozen=False, slots=True)
class SchemeConfigurationDefault:
    """Default scheme configuration used when only scheme settings are needed."""

    scheme_tolerance: Tolerance = field(default_factory=Tolerance)
    adaptive_scheme_safety: float = 0.8
    adaptive_scheme_min_factor: float = 0.1
    adaptive_scheme_max_factor: float = 5.0
    adaptive_scheme_error_exponent: float = 0.2
    adaptive_scheme_maximum_rejections: int | None = 64
    scheme_predictor: SchemePredictorLike | None = field(default_factory=SchemePredictorKnown)


__all__ = ["SchemeConfiguration", "SchemeConfigurationDefault"]
