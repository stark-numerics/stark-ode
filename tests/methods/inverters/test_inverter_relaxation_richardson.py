from __future__ import annotations

import pytest

from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.relaxation import InverterRelaxationRichardson
from stark import Configuration, Tolerance
from stark.diagnostics.monitor import MonitorInverter
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from tests.support import DummyScalarEntryOperator, DummyScalarTranslation


def test_richardson_solves_one_dimensional_scaled_request() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(2.0), size=1),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationRichardson[DummyScalarTranslation](
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=4),
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert inverter.defect.block is not None
    assert inverter.defect.block[0].value == pytest.approx(0.0)


def test_richardson_records_success_through_init_time_monitor() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(2.0), size=1),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationRichardson[DummyScalarTranslation](
        damping=0.5,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=4),
        monitor=monitor,
    )

    inverter(request, output)

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == "Richardson"
    assert solve.converged is True
    assert solve.iteration_count == 1
    assert solve.initial_residual == pytest.approx(6.0)
    assert solve.final_residual == pytest.approx(0.0)
    assert solve.failure_reason is None


def test_richardson_records_initial_acceptance() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(1.0), size=1),
        residual=Block([DummyScalarTranslation(2.0)]),
    )
    output = Block([DummyScalarTranslation(2.0)])
    inverter = InverterRelaxationRichardson[DummyScalarTranslation](
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0)),
        monitor=monitor,
    )

    inverter(request, output)

    solve = monitor.solves[0]
    assert solve.converged is True
    assert solve.iteration_count == 0
    assert solve.initial_residual == pytest.approx(0.0)
    assert solve.final_residual == pytest.approx(0.0)


def test_richardson_records_failure_when_budget_is_exhausted() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(DummyScalarEntryOperator(2.0), size=1),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationRichardson[DummyScalarTranslation](
        damping=0.25,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=1),
        monitor=monitor,
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(1.5)
    solve = monitor.solves[0]
    assert solve.converged is False
    assert solve.iteration_count == 1
    assert solve.initial_residual == pytest.approx(6.0)
    assert solve.final_residual == pytest.approx(3.0)
    assert solve.failure_reason == "maximum steps reached"


def test_richardson_rejects_non_positive_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        InverterRelaxationRichardson(damping=0.0)
