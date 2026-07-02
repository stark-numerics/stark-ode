from __future__ import annotations

import pytest

from stark.diagnostics.monitor import (
    Monitor,
    MonitorInverter,
    MonitorResolvent,
    MonitorScheme,
    MonitorSummary,
    MonitorInverterSummary,
    MonitorResolventSummary,
    MonitorSchemeSummary,
)


def test_monitor_scheme_records_fixed_steps() -> None:
    monitor = MonitorScheme()

    monitor.record_fixed_step("RK4", 0.25, 0.125)

    assert monitor.adaptive_steps == []
    assert len(monitor.fixed_steps) == 1

    step = monitor.fixed_steps[0]
    assert step.scheme == "RK4"
    assert step.t_start == pytest.approx(0.25)
    assert step.t_end == pytest.approx(0.375)
    assert step.accepted_dt == pytest.approx(0.125)


def test_monitor_scheme_records_adaptive_steps() -> None:
    monitor = MonitorScheme()

    monitor.record_adaptive_step("RKCK", 0.0, 0.1, 0.075, 0.05, 0.8, 2)

    assert monitor.fixed_steps == []
    assert len(monitor.adaptive_steps) == 1

    step = monitor.adaptive_steps[0]
    assert step.scheme == "RKCK"
    assert step.t_start == pytest.approx(0.0)
    assert step.t_end == pytest.approx(0.075)
    assert step.proposed_dt == pytest.approx(0.1)
    assert step.accepted_dt == pytest.approx(0.075)
    assert step.next_dt == pytest.approx(0.05)
    assert step.error_ratio == pytest.approx(0.8)
    assert step.rejection_count == 2


def test_monitor_clear_clears_scheme_records() -> None:
    monitor = Monitor()
    monitor.scheme.record_fixed_step("Euler", 0.0, 0.1)
    monitor.scheme.record_adaptive_step("RKCK", 0.1, 0.2, 0.2, 0.0, 0.0, 0)
    monitor.resolvent.record_solve("Picard", 0.1, 1, 3, 1.0e-9, 1.0, True)
    monitor.inverter.record_solve("GMRES", True, 2, 1.0, 1.0e-9, None)

    monitor.clear()

    assert monitor.scheme.fixed_steps == []
    assert monitor.scheme.adaptive_steps == []
    assert monitor.resolvent.solves == []
    assert monitor.inverter.solves == []


def test_monitor_default_and_explicit_channels_are_independent() -> None:
    default = Monitor()
    constructed = Monitor()
    existing_scheme = MonitorScheme()
    existing_resolvent = MonitorResolvent()
    existing_inverter = MonitorInverter()
    manual = Monitor(scheme=existing_scheme)
    manual_with_resolvent = Monitor(scheme=existing_scheme, resolvent=existing_resolvent)
    manual_with_all = Monitor(
        scheme=existing_scheme,
        resolvent=existing_resolvent,
        inverter=existing_inverter,
    )

    assert isinstance(default.scheme, MonitorScheme)
    assert isinstance(default.resolvent, MonitorResolvent)
    assert isinstance(default.inverter, MonitorInverter)
    assert isinstance(constructed.scheme, MonitorScheme)
    assert isinstance(constructed.resolvent, MonitorResolvent)
    assert isinstance(constructed.inverter, MonitorInverter)
    assert constructed.scheme is not default.scheme
    assert constructed.resolvent is not default.resolvent
    assert constructed.inverter is not default.inverter
    assert manual.scheme is existing_scheme
    assert manual_with_resolvent.resolvent is existing_resolvent
    assert manual_with_all.inverter is existing_inverter


def test_monitor_resolvent_records_solves() -> None:
    monitor = MonitorResolvent()

    monitor.record_solve("Picard", 0.125, 1, 3, 1.0e-9, 0.5, True)

    assert len(monitor.solves) == 1

    solve = monitor.solves[0]
    assert solve.resolvent == "Picard"
    assert solve.alpha == pytest.approx(0.125)
    assert solve.block_size == 1
    assert solve.iteration_count == 3
    assert solve.error == pytest.approx(1.0e-9)
    assert solve.scale == pytest.approx(0.5)
    assert solve.converged is True


def test_monitor_scheme_summary_reports_fixed_step_ranges() -> None:
    monitor = MonitorScheme()
    monitor.record_fixed_step("Euler", 0.0, 0.1)
    monitor.record_fixed_step("Euler", 0.1, 0.3)

    summary = monitor.summary()

    assert isinstance(summary, MonitorSchemeSummary)
    assert summary.step_count == 2
    assert summary.fixed_step_count == 2
    assert summary.adaptive_step_count == 0
    assert summary.accepted_dt_min == pytest.approx(0.1)
    assert summary.accepted_dt_median == pytest.approx(0.2)
    assert summary.accepted_dt_max == pytest.approx(0.3)
    assert summary.adaptive_rejection_count is None
    assert summary.adaptive_rejection_max is None
    assert summary.adaptive_error_ratio_min is None
    assert summary.adaptive_error_ratio_median is None
    assert summary.adaptive_error_ratio_max is None


def test_monitor_scheme_summary_reports_adaptive_step_ranges() -> None:
    monitor = MonitorScheme()
    monitor.record_fixed_step("Euler", 0.0, 0.1)
    monitor.record_adaptive_step("RKCK", 0.1, 0.2, 0.2, 0.3, 0.4, 0)
    monitor.record_adaptive_step("RKCK", 0.3, 0.3, 0.4, 0.5, 0.8, 2)

    summary = monitor.summary()

    assert summary.step_count == 3
    assert summary.fixed_step_count == 1
    assert summary.adaptive_step_count == 2
    assert summary.accepted_dt_min == pytest.approx(0.1)
    assert summary.accepted_dt_median == pytest.approx(0.2)
    assert summary.accepted_dt_max == pytest.approx(0.4)
    assert summary.adaptive_rejection_count == 2
    assert summary.adaptive_rejection_max == 2
    assert summary.adaptive_error_ratio_min == pytest.approx(0.4)
    assert summary.adaptive_error_ratio_median == pytest.approx(0.6)
    assert summary.adaptive_error_ratio_max == pytest.approx(0.8)


def test_empty_monitor_summary_uses_none_for_unavailable_ranges() -> None:
    monitor = Monitor()

    summary = monitor.summary()

    assert isinstance(summary, MonitorSummary)
    assert isinstance(summary.resolvent, MonitorResolventSummary)
    assert isinstance(summary.inverter, MonitorInverterSummary)
    assert summary.scheme.step_count == 0
    assert summary.scheme.fixed_step_count == 0
    assert summary.scheme.adaptive_step_count == 0
    assert summary.scheme.accepted_dt_min is None
    assert summary.scheme.accepted_dt_median is None
    assert summary.scheme.accepted_dt_max is None
    assert summary.scheme.adaptive_rejection_count is None
    assert summary.scheme.adaptive_rejection_max is None
    assert summary.scheme.adaptive_error_ratio_min is None
    assert summary.scheme.adaptive_error_ratio_median is None
    assert summary.scheme.adaptive_error_ratio_max is None
    assert summary.resolvent.solve_count == 0
    assert summary.resolvent.failure_count == 0
    assert summary.resolvent.iteration_min is None
    assert summary.resolvent.iteration_median is None
    assert summary.resolvent.iteration_max is None
    assert summary.resolvent.error_min is None
    assert summary.resolvent.error_median is None
    assert summary.resolvent.error_max is None
    assert summary.inverter.solve_count == 0
    assert summary.inverter.failure_count == 0
    assert summary.inverter.iteration_min is None
    assert summary.inverter.iteration_median is None
    assert summary.inverter.iteration_max is None
    assert summary.inverter.initial_residual_min is None
    assert summary.inverter.initial_residual_median is None
    assert summary.inverter.initial_residual_max is None
    assert summary.inverter.final_residual_min is None
    assert summary.inverter.final_residual_median is None
    assert summary.inverter.final_residual_max is None


def test_monitor_resolvent_summary_reports_iteration_and_error_ranges() -> None:
    monitor = MonitorResolvent()
    monitor.record_solve("Picard", 0.1, 1, 1, 1.0e-8, 1.0, True)
    monitor.record_solve("Picard", 0.2, 1, 3, 1.0e-6, 1.0, False)

    summary = monitor.summary()

    assert summary.solve_count == 2
    assert summary.failure_count == 1
    assert summary.iteration_min == 1
    assert summary.iteration_median == pytest.approx(2.0)
    assert summary.iteration_max == 3
    assert summary.error_min == pytest.approx(1.0e-8)
    assert summary.error_median == pytest.approx(5.05e-7)
    assert summary.error_max == pytest.approx(1.0e-6)
