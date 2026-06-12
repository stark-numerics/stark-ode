from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark.core.block import Block
from stark.core.block.operator import BlockOperatorDiagonal
from stark.methods.inverters.relaxation import InverterRelaxationRichardson
from stark import Configuration, Tolerance
from stark.diagnostics.monitor import MonitorInverter
from stark.methods.resolvents.requests.inverter import ResolventInverterRequest


@dataclass(slots=True)
class TranslationScalar:
    value: float = 0.0

    def __call__(self, origin, result) -> None:
        result.value = origin.value + self.value

    def norm(self) -> float:
        return abs(self.value)

    def __add__(self, other: "TranslationScalar") -> "TranslationScalar":
        return TranslationScalar(self.value + other.value)

    def __rmul__(self, scalar: float) -> "TranslationScalar":
        return TranslationScalar(scalar * self.value)


def scale_by_two(source: TranslationScalar, target: TranslationScalar) -> None:
    target.value = 2.0 * source.value


def identity(source: TranslationScalar, target: TranslationScalar) -> None:
    target.value = source.value


def test_richardson_solves_one_dimensional_scaled_request() -> None:
    request = ResolventInverterRequest(
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationRichardson[TranslationScalar](
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
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationRichardson[TranslationScalar](
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
        operator=BlockOperatorDiagonal.repeated(identity, size=1),
        residual=Block([TranslationScalar(2.0)]),
    )
    output = Block([TranslationScalar(2.0)])
    inverter = InverterRelaxationRichardson[TranslationScalar](
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
        operator=BlockOperatorDiagonal.repeated(scale_by_two, size=1),
        residual=Block([TranslationScalar(6.0)]),
    )
    output = Block([TranslationScalar(0.0)])
    inverter = InverterRelaxationRichardson[TranslationScalar](
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
