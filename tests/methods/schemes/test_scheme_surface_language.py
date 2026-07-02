from __future__ import annotations

from stark import Configuration
from stark.methods.schemes.explicit.fixed.euler import SchemeEuler
from stark.methods.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.methods.schemes.implicit.fixed.backward_euler import SchemeBackwardEuler
from stark.methods.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.methods.schemes.execution.step_control import SchemeStepControl


def test_snapshot_state_is_a_real_scheme_method() -> None:
    assert callable(SchemeEuler.__dict__["snapshot_state"])
    assert callable(SchemeBackwardEuler.__dict__["snapshot_state"])
    assert callable(SchemeIMEXEuler.__dict__["snapshot_state"])


def test_adaptive_monitoring_is_named_as_step_monitoring() -> None:
    assert hasattr(SchemeCashKarp, "call_monitored")


def test_adaptive_step_control_is_not_executor_lifecycle_state() -> None:
    control = SchemeStepControl(Configuration())
    assert not hasattr(control, "assign_executor")
    assert not hasattr(control, "unassign_executor")
    assert not hasattr(control, "runtime_bound")
