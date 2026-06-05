from __future__ import annotations

from dataclasses import dataclass

import pytest

from stark import Configuration, Tolerance
from stark.contracts import InverterRequest
from stark.inverters.support import (
    InverterDescriptor,
    with_inverter_monitoring,
)
from stark.monitor import MonitorInverter
from stark.resolvents.requests.inverter import ResolventInverterRequest


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


@dataclass(slots=True)
class BlockScalar:
    items: list[TranslationScalar]

    @property
    def size(self) -> int:
        return len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> TranslationScalar:
        return self.items[index]

    def __setitem__(self, index: int, value: TranslationScalar) -> None:
        self.items[index] = value

    def replace(self, other: "BlockScalar") -> None:
        for index, item in enumerate(other.items):
            self.items[index].value = item.value

    def norm(self) -> float:
        return sum(item.value * item.value for item in self.items) ** 0.5


@dataclass(slots=True)
class OperatorScalar:
    scale: float

    @property
    def size(self) -> int:
        return 1

    def reset(self) -> None:
        pass

    def __call__(self, source: BlockScalar, target: BlockScalar) -> None:
        target[0].value = self.scale * source[0].value


def accepts_inverter_request(request: InverterRequest[TranslationScalar]) -> float:
    return request.residual.norm()


def test_resolvent_inverter_request_satisfies_inverter_request_shape() -> None:
    request = ResolventInverterRequest(
        operator=OperatorScalar(2.0),
        residual=BlockScalar([TranslationScalar(4.0)]),
    )

    assert accepts_inverter_request(request) == pytest.approx(4.0)


@with_inverter_monitoring
class InverterMonitoredDummy:
    descriptor = InverterDescriptor("Dummy", "Dummy inverter")

    def __init__(self, monitor=None) -> None:
        self.monitor = monitor


def test_inverter_monitoring_records_through_init_time_monitor() -> None:
    monitor = MonitorInverter()
    inverter = InverterMonitoredDummy(monitor=monitor)

    inverter.record_solve(
        converged=True,
        iteration_count=3,
        initial_residual=1.0,
        final_residual=1.0e-8,
    )

    assert len(monitor.solves) == 1
    solve = monitor.solves[0]
    assert solve.inverter == "Dummy"
    assert solve.converged is True
    assert solve.iteration_count == 3
    assert solve.initial_residual == pytest.approx(1.0)
    assert solve.final_residual == pytest.approx(1.0e-8)
    assert solve.failure_reason is None


def test_inverter_monitoring_is_noop_without_monitor() -> None:
    inverter = InverterMonitoredDummy()

    inverter.record_solve(
        converged=True,
        iteration_count=None,
        initial_residual=None,
        final_residual=None,
    )


@pytest.mark.parametrize("maximum_steps", [0, -1])
def test_inverter_budget_rejects_non_positive_step_count(maximum_steps: int) -> None:
    with pytest.raises(ValueError, match="inverter_maximum_steps"):
        Configuration(inverter_maximum_steps=maximum_steps)


def test_inverter_tolerance_uses_Configuration_tolerance_semantics() -> None:
    tolerance = Tolerance(atol=1.0e-3, rtol=1.0e-2)

    assert tolerance.bound(2.0) == pytest.approx(2.1e-2)
    assert tolerance.accepts(2.0e-2, 2.0)
    assert not tolerance.accepts(3.0e-2, 2.0)
