"""Basic smoke tests for the project scaffold."""

import importlib

from stark import Marcher, Auditor, Integrator, Interval, Regulator, Tolerance
from stark.scheme_butcher_tableau import ButcherTableau


def test_package_imports() -> None:
    """The top-level package should import cleanly."""
    assert importlib.import_module("stark") is not None


def test_marcher_module_imports() -> None:
    """The marcher module should exist and import cleanly."""
    assert importlib.import_module("stark.marcher") is not None


def test_audit_module_imports() -> None:
    """The audit module should exist and import cleanly."""
    assert importlib.import_module("stark.audit") is not None


def test_control_module_imports() -> None:
    """The control module should exist and import cleanly."""
    assert importlib.import_module("stark.control") is not None


def test_integrate_module_imports() -> None:
    """The integrate module should exist and import cleanly."""
    assert importlib.import_module("stark.integrate") is not None


def test_scheme_library_imports() -> None:
    """The scheme library should expose aggregate and grouped public imports."""
    scheme_library = importlib.import_module("stark.scheme_library")
    adaptive = importlib.import_module("stark.scheme_library.adaptive")
    fixed_step = importlib.import_module("stark.scheme_library.fixed_step")

    assert scheme_library.SchemeDormandPrince is adaptive.SchemeDormandPrince
    assert scheme_library.SchemeCashKarp is adaptive.SchemeCashKarp
    assert scheme_library.SchemeTsitouras5 is adaptive.SchemeTsitouras5
    assert scheme_library.SchemeRK4 is fixed_step.SchemeRK4
    assert scheme_library.SchemeEuler is fixed_step.SchemeEuler


class MinimalScheme:
    def __call__(self, interval: Interval, state: object, tolerance: Tolerance) -> float:
        del interval, state, tolerance
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state

    def set_apply_delta_safety(self, enabled: bool) -> None:
        del enabled


def test_core_objects_have_readable_representations() -> None:
    interval = Interval(0.0, 0.1, 1.0)
    tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    regulator = Regulator()
    tableau = ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E")
    marcher = Marcher(MinimalScheme(), tolerance)
    auditor = Auditor(interval=interval, marcher=marcher, snapshots=True, exercise=False)

    assert repr(interval) == "Interval(present=0.0, step=0.1, stop=1.0)"
    assert str(interval) == "[0, 1] step=0.1"
    assert repr(tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert str(tolerance) == "atol=1e-08, rtol=1e-06"
    assert "Regulator" in repr(regulator)
    assert "safety=" in str(regulator)
    assert repr(tableau) == "ButcherTableau(stages=1, order=1, embedded_order=None, name='E')"
    assert str(Integrator()) == "STARK integrator (safe mode)"
    assert repr(marcher) == "Marcher(scheme='MinimalScheme', tolerance=Tolerance(atol=1e-08, rtol=1e-06), apply_delta_safety=True)"
    assert str(marcher) == "Marcher MinimalScheme with atol=1e-08, rtol=1e-06"
    assert "Auditor(status=" in repr(auditor)


def test_benchmark_packages_import() -> None:
    assert importlib.import_module("benchmarks.brusselator_2d.common") is not None
    assert importlib.import_module("benchmarks.fput.common") is not None
