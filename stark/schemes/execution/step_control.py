from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable

from stark.core import Configuration
from stark.schemes.configuration import SchemeConfiguration, SchemeConfigurationDefault
from stark.contracts import IntervalLike, State

ErrorRatio = Callable[[float, float], float]
ErrorBound = Callable[[float], float]

@dataclass(frozen=True, slots=True)
class SchemeStepAdaptiveAdvanceReport:
    """Compact accepted-advance report for adaptive schemes."""

    accepted_dt: float
    t_start: float
    proposed_dt: float
    next_dt: float
    error_ratio: float
    rejection_count: int

    @property
    def t_end(self) -> float:
        return self.t_start + self.accepted_dt


@dataclass(frozen=True, slots=True)
class SchemeStepAdaptiveProposal:
    """Initial step proposal data for one adaptive advance attempt."""

    remaining: float
    dt: float
    proposed_dt: float
    t_start: float


class SchemeStepControl:
    """Small adaptive step-control support object.

    Concrete schemes own their call routing. This object keeps adaptive policy,
    tolerance helpers, and the latest advance report.
    """

    __slots__ = (
        "_bound",
        "_configuration",
        "_max_factor",
        "_min_factor",
        "_ratio",
        "_report",
        "_safety",
        "_error_exponent",
    )

    def __init__(self, configuration: SchemeConfiguration) -> None:
        self._configuration = configuration
        self._safety = configuration.adaptive_scheme_safety
        self._min_factor = configuration.adaptive_scheme_min_factor
        self._max_factor = configuration.adaptive_scheme_max_factor
        self._error_exponent = configuration.adaptive_scheme_error_exponent
        self._ratio = configuration.scheme_tolerance.ratio
        self._bound = configuration.scheme_tolerance.bound
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=0.0,
            t_start=0.0,
            proposed_dt=0.0,
            next_dt=0.0,
            error_ratio=0.0,
            rejection_count=0,
        )

    @classmethod
    def from_configuration(cls, configuration: SchemeConfiguration) -> SchemeStepControl:
        return cls(configuration)

    @property
    def configuration(self) -> SchemeConfiguration:
        return self._configuration

    @property
    def ratio(self) -> ErrorRatio:
        return self._ratio

    @property
    def bound(self) -> ErrorBound:
        return self._bound

    def factor(self, error_ratio: float) -> float:
        if error_ratio == 0.0:
            return self._max_factor
        factor = self._safety * (1.0 / error_ratio) ** self._error_exponent
        return min(self._max_factor, max(self._min_factor, factor))

    def propose_step(self, interval: IntervalLike) -> SchemeStepAdaptiveProposal:
        remaining = interval.stop - interval.present
        if remaining <= 0.0:
            return SchemeStepAdaptiveProposal(
                remaining=remaining,
                dt=0.0,
                proposed_dt=0.0,
                t_start=interval.present,
            )

        dt = interval.step if interval.step <= remaining else remaining
        return SchemeStepAdaptiveProposal(
            remaining=remaining,
            dt=dt,
            proposed_dt=dt,
            t_start=interval.present,
        )

    def rejected_step(
        self,
        dt: float,
        error_ratio: float,
        remaining: float,
        label: str,
    ) -> float:
        dt *= self.factor(error_ratio)
        if dt <= 0.0:
            raise RuntimeError(f"{label} step size underflowed to zero.")
        return remaining if dt > remaining else dt

    def accepted_next_step(
        self,
        accepted_dt: float,
        error_ratio: float,
        remaining_after: float,
    ) -> float:
        if remaining_after <= 0.0:
            return 0.0
        next_step = accepted_dt * self.factor(error_ratio)
        return remaining_after if next_step > remaining_after else next_step

    def record_stopped(self, interval: IntervalLike) -> SchemeStepAdaptiveAdvanceReport:
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=0.0,
            t_start=interval.present,
            proposed_dt=0.0,
            next_dt=0.0,
            error_ratio=0.0,
            rejection_count=0,
        )
        return self._report

    def record_accepted(
        self,
        *,
        accepted_dt: float,
        t_start: float,
        proposed_dt: float,
        next_dt: float,
        error_ratio: float,
        rejection_count: int,
    ) -> SchemeStepAdaptiveAdvanceReport:
        self._report = SchemeStepAdaptiveAdvanceReport(
            accepted_dt=accepted_dt,
            t_start=t_start,
            proposed_dt=proposed_dt,
            next_dt=next_dt,
            error_ratio=error_ratio,
            rejection_count=rejection_count,
        )
        return self._report

    def report(self) -> SchemeStepAdaptiveAdvanceReport:
        return self._report


def default_adaptive_error_exponent(scheme) -> float:
    default_adaptivity = getattr(type(scheme), "default_adaptivity", None)
    if default_adaptivity is None:
        return Configuration().adaptive_scheme_error_exponent
    value = default_adaptivity()
    return value.error_exponent if hasattr(value, "error_exponent") else float(value)


def default_scheme_configuration(scheme) -> Configuration:
    return replace(
        Configuration(),
        adaptive_scheme_error_exponent=default_adaptive_error_exponent(scheme),
    )



__all__ = [
    "ErrorBound",
    "ErrorRatio",
    "SchemeStepAdaptiveProposal",
    "SchemeStepAdaptiveAdvanceReport",
    "SchemeStepControl",
    "default_adaptive_error_exponent",
    "default_scheme_configuration",
]
