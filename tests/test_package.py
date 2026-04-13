"""Basic smoke tests for the project scaffold."""

import importlib

from stark import (
    Marcher,
    Auditor,
    Block,
    BlockOperator,
    Integrator,
    InverterPolicy,
    InverterTolerance,
    Interval,
    InverterDescriptor,
    Regulator,
    ResolverPolicy,
    ResolverTolerance,
    ResolverDescriptor,
    Safety,
    SchemeTolerance,
    Tolerance,
)
from stark.butcher_tableau import ButcherTableau


def test_package_imports() -> None:
    """The top-level package should import cleanly."""
    assert importlib.import_module("stark") is not None


def test_marcher_module_imports() -> None:
    """The marcher module should exist and import cleanly."""
    assert importlib.import_module("stark.marcher") is not None


def test_audit_module_imports() -> None:
    """The audit module should exist and import cleanly."""
    assert importlib.import_module("stark.audit") is not None


def test_regulator_module_imports() -> None:
    """The regulator module should exist and import cleanly."""
    assert importlib.import_module("stark.regulator") is not None


def test_integrate_module_imports() -> None:
    """The integrate module should exist and import cleanly."""
    assert importlib.import_module("stark.integrate") is not None


def test_inverter_library_imports() -> None:
    """The inverter library should import cleanly."""
    inverter_library = importlib.import_module("stark.inverter_library")

    assert inverter_library.InverterGMRES is not None
    assert inverter_library.InverterFGMRES is not None
    assert inverter_library.InverterBiCGStab is not None


def test_resolver_library_imports() -> None:
    """The resolver library should import cleanly."""
    resolver_library = importlib.import_module("stark.resolver_library")

    assert resolver_library.ResolverAnderson is not None
    assert resolver_library.ResolverBroyden is not None
    assert resolver_library.ResolverPicard is not None
    assert resolver_library.ResolverNewton is not None


def test_scheme_library_imports() -> None:
    """The scheme library should expose aggregate and grouped public imports."""
    scheme_library = importlib.import_module("stark.scheme_library")
    adaptive = importlib.import_module("stark.scheme_library.adaptive")
    adaptive_implicit = importlib.import_module("stark.scheme_library.adaptive_implicit")
    fixed_step = importlib.import_module("stark.scheme_library.fixed_step")
    implicit = importlib.import_module("stark.scheme_library.implicit")

    assert scheme_library.SchemeDormandPrince is adaptive.SchemeDormandPrince
    assert scheme_library.SchemeCashKarp is adaptive.SchemeCashKarp
    assert scheme_library.SchemeTsitouras5 is adaptive.SchemeTsitouras5
    assert scheme_library.SchemeBDF2 is adaptive_implicit.SchemeBDF2
    assert scheme_library.SchemeKvaerno3 is adaptive_implicit.SchemeKvaerno3
    assert scheme_library.SchemeKvaerno4 is adaptive_implicit.SchemeKvaerno4
    assert scheme_library.SchemeSDIRK21 is adaptive_implicit.SchemeSDIRK21
    assert scheme_library.SchemeRK4 is fixed_step.SchemeRK4
    assert scheme_library.SchemeEuler is fixed_step.SchemeEuler
    assert scheme_library.SchemeBackwardEuler is implicit.SchemeBackwardEuler


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
    block = Block([])
    block_operator = BlockOperator([])
    inverter_policy = InverterPolicy()
    inverter_tolerance = InverterTolerance(atol=1.0e-8, rtol=1.0e-6)
    inverter_descriptor = InverterDescriptor("GMRES", "Restarted GMRES")
    resolver_policy = ResolverPolicy()
    resolver_tolerance = ResolverTolerance(atol=1.0e-8, rtol=1.0e-6)
    resolver_descriptor = ResolverDescriptor("Picard", "Picard Iteration")
    safety = Safety()
    tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    scheme_tolerance = SchemeTolerance(atol=1.0e-8, rtol=1.0e-6)
    regulator = Regulator()
    tableau = ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E")
    marcher = Marcher(MinimalScheme(), tolerance)
    auditor = Auditor(interval=interval, marcher=marcher, snapshots=True, exercise=False)

    assert repr(interval) == "Interval(present=0.0, step=0.1, stop=1.0)"
    assert str(interval) == "[0, 1] step=0.1"
    assert repr(block) == "Block(size=0)"
    assert str(block) == "block[0]"
    assert repr(block_operator) == "BlockOperator(size=0)"
    assert str(block_operator) == "block operator[0]"
    assert repr(inverter_policy) == "InverterPolicy(max_iterations=32, restart=16, breakdown_tol=1e-30)"
    assert str(inverter_policy) == "max_iterations=32, restart=16, breakdown_tol=1e-30"
    assert repr(inverter_tolerance) == "InverterTolerance(atol=1e-08, rtol=1e-06)"
    assert repr(inverter_descriptor) == "InverterDescriptor(short_name='GMRES', full_name='Restarted GMRES')"
    assert repr(resolver_policy) == "ResolverPolicy(max_iterations=16)"
    assert str(resolver_policy) == "max_iterations=16"
    assert repr(resolver_tolerance) == "ResolverTolerance(atol=1e-08, rtol=1e-06)"
    assert repr(resolver_descriptor) == "ResolverDescriptor(short_name='Picard', full_name='Picard Iteration')"
    assert repr(safety) == "Safety(progress=True, block_sizes=True, apply_delta=True)"
    assert repr(tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert repr(scheme_tolerance) == "SchemeTolerance(atol=1e-08, rtol=1e-06)"
    assert str(tolerance) == "atol=1e-08, rtol=1e-06"
    assert "Regulator" in repr(regulator)
    assert "safety=" in str(regulator)
    assert repr(tableau) == "ButcherTableau(stages=1, order=1, embedded_order=None, name='E')"
    assert str(Integrator()) == "STARK integrator (safe mode)"
    assert repr(marcher) == "Marcher(scheme='MinimalScheme', tolerance=Tolerance(atol=1e-08, rtol=1e-06), safety=Safety(progress=True, block_sizes=True, apply_delta=True))"
    assert str(marcher) == "Marcher MinimalScheme with atol=1e-08, rtol=1e-06"
    assert "Auditor(status=" in repr(auditor)


def test_benchmark_packages_import() -> None:
    assert importlib.import_module("benchmarks.brusselator_2d.common") is not None
    assert importlib.import_module("benchmarks.fitzhugh_nagumo_1d.common") is not None
    assert importlib.import_module("benchmarks.fput.common") is not None

