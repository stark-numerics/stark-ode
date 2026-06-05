from __future__ import annotations

from stark import Configuration
from stark.schemes.explicit.fixed.euler import SchemeEuler
from stark.schemes.explicit.adaptive.cash_karp import SchemeCashKarp
from stark.schemes.implicit.fixed.backward_euler import SchemeBackwardEuler
from stark.schemes.imex.fixed.euler import SchemeIMEXEuler
from stark.schemes.execution.step_control import SchemeStepControl
from stark.schemes.explicit._support import explicit_snapshot_state
from stark.schemes.implicit._support import implicit_snapshot_state
from stark.schemes.imex._support import imex_snapshot_state


def test_workspace_support_methods_are_visible_class_imports() -> None:
    assert SchemeEuler.snapshot_state is explicit_snapshot_state
    assert SchemeBackwardEuler.snapshot_state is implicit_snapshot_state
    assert SchemeIMEXEuler.snapshot_state is imex_snapshot_state


def test_adaptive_monitoring_is_named_as_step_monitoring() -> None:
    assert hasattr(SchemeCashKarp, "call_monitored")


def test_adaptive_step_control_is_not_executor_lifecycle_state() -> None:
    control = SchemeStepControl(Configuration())
    assert not hasattr(control, "assign_executor")
    assert not hasattr(control, "unassign_executor")
    assert not hasattr(control, "runtime_bound")
