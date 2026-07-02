from __future__ import annotations

import pytest

from stark import Interval, Monitor
from stark.core.block import Block
from stark.methods.resolvents import ResolventPicard
from stark.methods.schemes.request import SchemeResolventRequest
from tests.support import (
    DummyScalarAllocator,
    DummyScalarState,
    DummyScalarTranslation,
    dummy_zero_rhs,
)


def test_picard_records_resolvent_solve_when_monitor_assigned() -> None:
    monitor = Monitor()
    resolvent = ResolventPicard(DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = DummyScalarState()
    out = Block([DummyScalarTranslation()])
    problem = SchemeResolventRequest(dummy_zero_rhs, interval, state, None, 0.1)

    resolvent.assign_monitor(monitor.resolvent)
    resolvent(problem, out)

    assert len(monitor.resolvent.solves) == 1

    solve = monitor.resolvent.solves[0]
    assert solve.resolvent == "Picard"
    assert solve.alpha == pytest.approx(0.1)
    assert solve.block_size == 1
    assert solve.iteration_count == 0
    assert solve.error == pytest.approx(0.0)
    assert solve.scale == pytest.approx(0.0)
    assert solve.converged is True


def test_picard_unassign_monitor_restores_unmonitored_solves() -> None:
    monitor = Monitor()
    resolvent = ResolventPicard(DummyScalarAllocator())
    interval = Interval(present=0.0, step=0.1, stop=1.0)
    state = DummyScalarState()
    out = Block([DummyScalarTranslation()])
    problem = SchemeResolventRequest(dummy_zero_rhs, interval, state, None, 0.1)

    resolvent.assign_monitor(monitor.resolvent)
    resolvent.unassign_monitor()
    resolvent(problem, out)

    assert monitor.resolvent.solves == []
