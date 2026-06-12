from __future__ import annotations

import pytest

from stark.diagnostics.monitor import MonitorInverter, MonitorInverterSummary


def test_monitor_inverter_records_successful_solve() -> None:
    monitor = MonitorInverter()

    monitor.record_solve("GMRES", True, 4, 1.0e-2, 1.0e-9, None)

    assert len(monitor.solves) == 1

    solve = monitor.solves[0]
    assert solve.inverter == "GMRES"
    assert solve.converged is True
    assert solve.iteration_count == 4
    assert solve.initial_residual == pytest.approx(1.0e-2)
    assert solve.final_residual == pytest.approx(1.0e-9)
    assert solve.failure_reason is None


def test_monitor_inverter_records_failed_solve() -> None:
    monitor = MonitorInverter()

    monitor.record_solve("BiCGStab", False, 8, 1.0, 1.0e-3, "vanishing omega")

    solve = monitor.solves[0]
    assert solve.inverter == "BiCGStab"
    assert solve.converged is False
    assert solve.iteration_count == 8
    assert solve.initial_residual == pytest.approx(1.0)
    assert solve.final_residual == pytest.approx(1.0e-3)
    assert solve.failure_reason == "vanishing omega"


def test_monitor_inverter_accepts_unavailable_direct_solve_fields() -> None:
    monitor = MonitorInverter()

    monitor.record_solve("DirectScalar", True, None, None, None, None)

    solve = monitor.solves[0]
    assert solve.iteration_count is None
    assert solve.initial_residual is None
    assert solve.final_residual is None
    assert solve.failure_reason is None


def test_monitor_inverter_summary_reports_iteration_and_residual_ranges() -> None:
    monitor = MonitorInverter()
    monitor.record_solve("GMRES", True, 2, 1.0e-1, 1.0e-8, None)
    monitor.record_solve("GMRES", False, 6, 1.0e0, 1.0e-4, "iteration limit")
    monitor.record_solve("DirectScalar", True, None, None, None, None)

    summary = monitor.summary()

    assert isinstance(summary, MonitorInverterSummary)
    assert summary.solve_count == 3
    assert summary.failure_count == 1
    assert summary.iteration_min == 2
    assert summary.iteration_median == pytest.approx(4.0)
    assert summary.iteration_max == 6
    assert summary.initial_residual_min == pytest.approx(1.0e-1)
    assert summary.initial_residual_median == pytest.approx(5.5e-1)
    assert summary.initial_residual_max == pytest.approx(1.0e0)
    assert summary.final_residual_min == pytest.approx(1.0e-8)
    assert summary.final_residual_median == pytest.approx(5.0005e-5)
    assert summary.final_residual_max == pytest.approx(1.0e-4)


def test_empty_monitor_inverter_summary_uses_none_for_unavailable_ranges() -> None:
    summary = MonitorInverter().summary()

    assert summary.solve_count == 0
    assert summary.failure_count == 0
    assert summary.iteration_min is None
    assert summary.iteration_median is None
    assert summary.iteration_max is None
    assert summary.initial_residual_min is None
    assert summary.initial_residual_median is None
    assert summary.initial_residual_max is None
    assert summary.final_residual_min is None
    assert summary.final_residual_median is None
    assert summary.final_residual_max is None


def test_monitor_inverter_clear_removes_records() -> None:
    monitor = MonitorInverter()
    monitor.record_solve("GMRES", True, 1, 1.0, 0.0, None)

    monitor.clear()

    assert monitor.solves == []
