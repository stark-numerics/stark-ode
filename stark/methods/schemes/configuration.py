from __future__ import annotations

from typing import Protocol
from dataclasses import dataclass, field
from stark.core.tolerance import Tolerance

@dataclass(frozen=False, slots=True)
class SchemeConfiguration(Protocol):
    scheme_tolerance: Tolerance
    adaptive_scheme_safety: float
    adaptive_scheme_min_factor: float
    adaptive_scheme_max_factor: float
    adaptive_scheme_error_exponent: float

@dataclass(frozen=False, slots=True)
class SchemeConfigurationDefault:
    scheme_tolerance: Tolerance = field(default_factory=Tolerance)
    adaptive_scheme_safety: float = 0.8
    adaptive_scheme_min_factor: float = 0.1
    adaptive_scheme_max_factor: float = 5.0
    adaptive_scheme_error_exponent: float = 0.2


__all__ = ["SchemeConfiguration", "SchemeConfigurationDefault"]
