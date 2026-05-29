from __future__ import annotations

from stark.schemes.explicit_fixed.euler import SchemeEuler
from stark.schemes.explicit_adaptive.cash_karp import SchemeCashKarp
from stark.schemes.implicit_fixed.backward_euler import SchemeBackwardEuler
from stark.schemes.imex_fixed.euler import SchemeIMEXEuler
from stark.schemes.support.adaptive import SchemeStepControl
from stark.schemes.support.explicit import (
    explicit_set_apply_delta_safety,
    explicit_snapshot_state,
)
from stark.schemes.support.implicit import (
    implicit_set_apply_delta_safety,
    implicit_snapshot_state,
)
from stark.schemes.support.imex import (
    imex_set_apply_delta_safety,
    imex_snapshot_state,
)


def test_workspace_support_methods_are_visible_class_imports() -> None:
    assert SchemeEuler.set_apply_delta_safety is explicit_set_apply_delta_safety
    assert SchemeEuler.snapshot_state is explicit_snapshot_state
    assert SchemeBackwardEuler.set_apply_delta_safety is implicit_set_apply_delta_safety
    assert SchemeBackwardEuler.snapshot_state is implicit_snapshot_state
    assert SchemeIMEXEuler.set_apply_delta_safety is imex_set_apply_delta_safety
    assert SchemeIMEXEuler.snapshot_state is imex_snapshot_state


def test_adaptive_monitoring_is_named_as_step_monitoring() -> None:
    assert hasattr(SchemeCashKarp, "call_monitored")


def test_adaptive_step_control_is_not_executor_lifecycle_state() -> None:
    control = SchemeStepControl()
    assert not hasattr(control, "assign_executor")
    assert not hasattr(control, "unassign_executor")
    assert not hasattr(control, "runtime_bound")
