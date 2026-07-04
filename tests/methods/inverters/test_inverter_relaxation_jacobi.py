from __future__ import annotations

from typing import cast

import pytest

from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.core.contracts import BlockOperatorEntryLike
from stark.methods.inverters.relaxation import InverterRelaxationJacobi
from stark import Configuration, Tolerance
from stark.diagnostics.monitor import MonitorInverter
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest
from tests.support import DummyScalarEntryOperator, DummyScalarTranslation


def invert_entry(
    operator: BlockOperatorEntryLike[DummyScalarTranslation],
    source: DummyScalarTranslation,
    target: DummyScalarTranslation,
) -> None:
    cast(DummyScalarEntryOperator, operator).inverse(source, target)


def test_jacobi_solves_diagonal_scaled_request_in_one_step() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([DummyScalarEntryOperator(2.0), DummyScalarEntryOperator(4.0)]),
        residual=Block([DummyScalarTranslation(6.0), DummyScalarTranslation(20.0)]),
    )
    output = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationJacobi[DummyScalarTranslation](
        invert_entry,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=2),
    )

    inverter(request, output)

    assert output[0].value == pytest.approx(3.0)
    assert output[1].value == pytest.approx(5.0)
    assert inverter.defect.block is not None
    assert inverter.defect.block.norm() == pytest.approx(0.0)


def test_jacobi_records_success_through_init_time_monitor() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([DummyScalarEntryOperator(2.0), DummyScalarEntryOperator(4.0)]),
        residual=Block([DummyScalarTranslation(6.0), DummyScalarTranslation(20.0)]),
    )
    output = Block([DummyScalarTranslation(0.0), DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationJacobi[DummyScalarTranslation](
        invert_entry,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0), inverter_maximum_steps=2),
        monitor=monitor,
    )

    inverter(request, output)

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == "Jacobi"
    assert solve.converged is True
    assert solve.iteration_count == 1
    assert solve.initial_residual == pytest.approx((6.0**2 + 20.0**2) ** 0.5)
    assert solve.final_residual == pytest.approx(0.0)
    assert solve.failure_reason is None


def test_jacobi_records_initial_acceptance() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([DummyScalarEntryOperator(2.0)]),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(3.0)])
    inverter = InverterRelaxationJacobi[DummyScalarTranslation](
        invert_entry,
        configuration=Configuration(inverter_tolerance=Tolerance(atol=1.0e-12, rtol=0.0)),
        monitor=monitor,
    )

    inverter(request, output)

    solve = monitor.solves[0]
    assert solve.converged is True
    assert solve.iteration_count == 0
    assert solve.initial_residual == pytest.approx(0.0)
    assert solve.final_residual == pytest.approx(0.0)


def test_jacobi_records_failure_when_budget_is_exhausted() -> None:
    monitor = MonitorInverter()
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal([DummyScalarEntryOperator(2.0)]),
        residual=Block([DummyScalarTranslation(6.0)]),
    )
    output = Block([DummyScalarTranslation(0.0)])
    inverter = InverterRelaxationJacobi[DummyScalarTranslation](
        invert_entry,
        damping=0.5,
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


def test_jacobi_rejects_non_positive_damping() -> None:
    with pytest.raises(ValueError, match="damping"):
        InverterRelaxationJacobi(invert_entry, damping=0.0)
