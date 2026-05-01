"""Basic smoke tests for the project scaffold."""

import importlib

from stark import Auditor, Executor, ImExDerivative, Integrator, Interval, Marcher, Regulator, Safety, SchemeTolerance, Tolerance
from stark.accelerators import Accelerator, AcceleratorAbsent
from stark.block.operator import BlockOperator
from stark.comparison import Comparator, ComparatorEntry, ComparatorProblem
from stark.contracts import Block, Resolvent
from stark.inverters.descriptor import InverterDescriptor
from stark.inverters.policy import InverterPolicy
from stark.inverters.tolerance import InverterTolerance
from stark.resolvents import (
    ResolventAnderson,
    ResolventBroyden,
    ResolventCoupledNewton,
    ResolventCoupledPicard,
    ResolventNewton,
    ResolventPicard,
)
from stark.resolvents.descriptor import ResolventDescriptor
from stark.resolvents.policy import ResolventPolicy
from stark.resolvents.tolerance import ResolventTolerance
from stark.schemes import (
    ARK324L2SA_TABLEAU,
    ARK436L2SA_TABLEAU,
    ARK437L2SA_TABLEAU,
    ARK548L2SAB_TABLEAU,
    ARK548L2SA_TABLEAU,
    BE_TABLEAU,
    CRANK_NICOLSON_TABLEAU,
    CROUZEIX_DIRK3_TABLEAU,
    GAUSS_LEGENDRE4_TABLEAU,
    IMEX_EULER_TABLEAU,
    IMPLICIT_MIDPOINT_TABLEAU,
    LOBATTO_IIIC4_TABLEAU,
    RADAU_IIA5_TABLEAU,
    SchemeBackwardEuler,
    SchemeCrankNicolson,
    SchemeCrouzeixDIRK3,
    SchemeGaussLegendre4,
    SchemeIMEXEuler,
    SchemeImplicitMidpoint,
    SchemeKennedyCarpenter32,
    SchemeKennedyCarpenter43_6,
    SchemeKennedyCarpenter43_7,
    SchemeKennedyCarpenter54,
    SchemeKennedyCarpenter54b,
    SchemeLobattoIIIC4,
    SchemeRadauIIA5,
)
from stark.schemes.tableau import ButcherTableau, ImExButcherTableau


def test_package_imports() -> None:
    """The top-level package should import cleanly."""
    assert importlib.import_module("stark") is not None
    assert Accelerator is not None
    assert ImExDerivative is not None
    assert Resolvent is not None


def test_comparator_module_imports() -> None:
    """The comparator module should exist and import cleanly."""
    assert importlib.import_module("stark.comparison") is not None


def test_marcher_module_imports() -> None:
    """The marcher module should exist and import cleanly."""
    assert importlib.import_module("stark.marcher") is not None


def test_audit_module_imports() -> None:
    """The audit module should exist and import cleanly."""
    assert importlib.import_module("stark.auditor") is not None


def test_regulator_module_imports() -> None:
    """The regulator module should exist and import cleanly."""
    assert importlib.import_module("stark.accelerators") is not None
    assert importlib.import_module("stark.execution.regulator") is not None
    assert importlib.import_module("stark.execution.adaptive_controller") is not None
    assert importlib.import_module("stark.execution.executor") is not None
    assert importlib.import_module("stark.execution.tolerance") is not None
    assert importlib.import_module("stark.execution.safety") is not None
    assert importlib.import_module("stark.schemes.tableau") is not None
    assert importlib.import_module("stark.machinery.stage_solve.workspace") is not None
    assert importlib.import_module("stark.machinery.stage_solve.workers") is not None
    assert importlib.import_module("stark.machinery.translation_algebra.linear_combine") is not None
    assert importlib.import_module("stark.schemes.display") is not None


def test_integrate_module_imports() -> None:
    """The integrate module should exist and import cleanly."""
    assert importlib.import_module("stark.integrate") is not None


def test_inverter_imports() -> None:
    """The inverter package should import cleanly."""
    inverters = importlib.import_module("stark.inverters")

    assert inverters.InverterGMRES is not None
    assert inverters.InverterFGMRES is not None
    assert inverters.InverterBiCGStab is not None


def test_resolvent_imports() -> None:
    """The resolvent package should expose methods and metadata cleanly."""
    resolvents = importlib.import_module("stark.resolvents")
    support = importlib.import_module("stark.resolvents.support")

    assert resolvents.ResolventPolicy is not None
    assert resolvents.ResolventTolerance is not None
    assert resolvents.ResolventError is not None
    assert resolvents.ResolventPicard is not None
    assert resolvents.ResolventNewton is not None
    assert resolvents.ResolventAnderson is not None
    assert resolvents.ResolventBroyden is not None
    assert resolvents.ResolventCoupledPicard is not None
    assert resolvents.ResolventCoupledNewton is not None
    assert support.StageResidual is not None


def test_scheme_imports() -> None:
    """The schemes package should expose aggregate and grouped public imports."""
    schemes = importlib.import_module("stark.schemes")
    adaptive = importlib.import_module("stark.schemes.explicit_adaptive")
    adaptive_implicit = importlib.import_module("stark.schemes.implicit_adaptive")
    fixed_step = importlib.import_module("stark.schemes.explicit_fixed")
    imex_adaptive = importlib.import_module("stark.schemes.imex_adaptive")
    imex_fixed = importlib.import_module("stark.schemes.imex_fixed")
    implicit = importlib.import_module("stark.schemes.implicit_fixed")

    assert schemes.SchemeDormandPrince is adaptive.SchemeDormandPrince
    assert schemes.SchemeCashKarp is adaptive.SchemeCashKarp
    assert schemes.SchemeTsitouras5 is adaptive.SchemeTsitouras5
    assert schemes.SchemeBDF2 is adaptive_implicit.SchemeBDF2
    assert schemes.SchemeKvaerno3 is adaptive_implicit.SchemeKvaerno3
    assert schemes.SchemeKvaerno4 is adaptive_implicit.SchemeKvaerno4
    assert schemes.SchemeSDIRK21 is adaptive_implicit.SchemeSDIRK21
    assert schemes.SchemeRK4 is fixed_step.SchemeRK4
    assert schemes.SchemeEuler is fixed_step.SchemeEuler
    assert schemes.SchemeIMEXEuler is imex_fixed.SchemeIMEXEuler
    assert schemes.SchemeKennedyCarpenter32 is imex_adaptive.SchemeKennedyCarpenter32
    assert schemes.SchemeKennedyCarpenter43_6 is imex_adaptive.SchemeKennedyCarpenter43_6
    assert schemes.SchemeKennedyCarpenter43_7 is imex_adaptive.SchemeKennedyCarpenter43_7
    assert schemes.SchemeKennedyCarpenter54 is imex_adaptive.SchemeKennedyCarpenter54
    assert schemes.SchemeKennedyCarpenter54b is imex_adaptive.SchemeKennedyCarpenter54b
    assert schemes.SchemeBackwardEuler is implicit.SchemeBackwardEuler
    assert schemes.SchemeImplicitMidpoint is implicit.SchemeImplicitMidpoint
    assert schemes.SchemeCrankNicolson is implicit.SchemeCrankNicolson
    assert schemes.SchemeCrouzeixDIRK3 is implicit.SchemeCrouzeixDIRK3
    assert schemes.SchemeGaussLegendre4 is implicit.SchemeGaussLegendre4
    assert schemes.SchemeLobattoIIIC4 is implicit.SchemeLobattoIIIC4
    assert schemes.SchemeRadauIIA5 is implicit.SchemeRadauIIA5


class MinimalScheme:
    def __call__(self, interval: Interval, state: object, executor: Executor) -> float:
        del interval, state, executor
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state

    def set_apply_delta_safety(self, enabled: bool) -> None:
        del enabled


class MinimalWorkbench:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, dst: object, src: object) -> None:
        del dst, src

    def allocate_translation(self) -> "MinimalTranslation":
        return MinimalTranslation()


class MinimalTranslation:
    def __call__(self, origin: object, result: object) -> None:
        del origin, result

    def norm(self) -> float:
        return 0.0

    def __add__(self, other: "MinimalTranslation") -> "MinimalTranslation":
        del other
        return MinimalTranslation()

    def __rmul__(self, scalar: float) -> "MinimalTranslation":
        del scalar
        return MinimalTranslation()

    @staticmethod
    def scale(out: "MinimalTranslation", a: float, x: "MinimalTranslation") -> "MinimalTranslation":
        del a, x
        return out

    @staticmethod
    def combine2(
        out: "MinimalTranslation",
        a0: float,
        x0: "MinimalTranslation",
        a1: float,
        x1: "MinimalTranslation",
    ) -> "MinimalTranslation":
        del a0, x0, a1, x1
        return out

    linear_combine = [scale, combine2]


class MinimalInverter:
    def bind(self, operator: object) -> None:
        del operator

    def __call__(self, out: Block, rhs: Block) -> None:
        del out, rhs


def test_core_objects_have_readable_representations() -> None:
    bakeoff_problem = ComparatorProblem("Dummy", lambda: object(), lambda: object(), lambda left, right: 0.0)
    bakeoff_entry = ComparatorEntry("Dummy", lambda: object())
    bakeoff = Comparator(
        bakeoff_problem,
        [bakeoff_entry, ComparatorEntry("Other", lambda: object())],
        repeats=1,
    )
    interval = Interval(0.0, 0.1, 1.0)
    block = Block([])
    block_operator = BlockOperator([])
    inverter_policy = InverterPolicy()
    inverter_tolerance = InverterTolerance(atol=1.0e-8, rtol=1.0e-6)
    inverter_descriptor = InverterDescriptor("GMRES", "Restarted GMRES")
    resolver_policy = ResolventPolicy()
    resolver_tolerance = ResolventTolerance(atol=1.0e-8, rtol=1.0e-6)
    resolver_descriptor = ResolventDescriptor("Picard", "Picard Iteration")
    safety = Safety()
    tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    scheme_tolerance = SchemeTolerance(atol=1.0e-8, rtol=1.0e-6)
    regulator = Regulator()
    tableau = ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E")
    imex_tableau = ImExButcherTableau(
        explicit=ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E"),
        implicit=ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="I"),
    )
    marcher = Marcher(MinimalScheme(), Executor(tolerance=tolerance))
    auditor = Auditor(interval=interval, marcher=marcher, snapshots=True, exercise=False)
    workbench = MinimalWorkbench()
    auto_picard = ResolventPicard(lambda interval, state, out: None, workbench, accelerator=AcceleratorAbsent())
    auto_coupled_picard = ResolventCoupledPicard(
        lambda interval, state, out: None,
        workbench,
        tableau=ButcherTableau(
            c=(0.5, 0.5),
            a=((0.25, 0.25), (0.25, 0.25)),
            b=(0.5, 0.5),
            order=2,
            short_name="DummyCoupled",
        ),
    )
    auto_anderson = ResolventAnderson(
        lambda interval, state, out: None,
        workbench,
        inner_product=lambda left, right: 0.0,
    )
    auto_broyden = ResolventBroyden(
        lambda interval, state, out: None,
        workbench,
        inner_product=lambda left, right: 0.0,
    )
    auto_newton = ResolventNewton(
        lambda interval, state, out: None,
        workbench,
        linearizer=lambda interval, out, state: setattr(out, "apply", lambda result, translation: None),
        inverter=MinimalInverter(),
    )
    auto_coupled_newton = ResolventCoupledNewton(
        lambda interval, state, out: None,
        workbench,
        tableau=ButcherTableau(
            c=(0.5, 0.5),
            a=((0.25, 0.25), (0.25, 0.25)),
            b=(0.5, 0.5),
            order=2,
            short_name="DummyCoupled",
        ),
        linearizer=lambda interval, out, state: setattr(out, "apply", lambda result, translation: None),
        inverter=MinimalInverter(),
    )

    assert repr(interval) == "Interval(present=0.0, step=0.1, stop=1.0)"
    assert repr(bakeoff_problem).startswith("ComparatorProblem(")
    assert repr(bakeoff_entry).startswith("ComparatorEntry(")
    assert "Comparator(" in repr(bakeoff)
    assert str(interval) == "[0, 1] step=0.1"
    assert repr(block) == "Block(size=0)"
    assert str(block) == "block[0]"
    assert repr(block_operator) == "BlockOperator(size=0)"
    assert str(block_operator) == "block operator[0]"
    assert repr(inverter_policy) == "InverterPolicy(max_iterations=32, restart=16, breakdown_tol=1e-30)"
    assert str(inverter_policy) == "max_iterations=32, restart=16, breakdown_tol=1e-30"
    assert repr(inverter_tolerance) == "InverterTolerance(atol=1e-08, rtol=1e-06)"
    assert repr(inverter_descriptor) == "InverterDescriptor(short_name='GMRES', full_name='Restarted GMRES')"
    assert repr(resolver_policy) == "ResolventPolicy(max_iterations=16)"
    assert str(resolver_policy) == "max_iterations=16"
    assert repr(resolver_tolerance) == "ResolventTolerance(atol=1e-08, rtol=1e-06)"
    assert repr(resolver_descriptor) == "ResolventDescriptor(short_name='Picard', full_name='Picard Iteration')"
    assert repr(safety) == "Safety(progress=True, block_sizes=True, apply_delta=True)"
    assert repr(tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert repr(scheme_tolerance) == "SchemeTolerance(atol=1e-08, rtol=1e-06)"
    assert str(tolerance) == "atol=1e-08, rtol=1e-06"
    assert "Regulator" in repr(regulator)
    assert "safety=" in str(regulator)
    assert repr(tableau) == "ButcherTableau(stages=1, order=1, embedded_order=None, name='E')"
    assert "ImExButcherTableau" in repr(imex_tableau)
    assert IMEX_EULER_TABLEAU is not None
    assert BE_TABLEAU is not None
    assert IMPLICIT_MIDPOINT_TABLEAU is not None
    assert CRANK_NICOLSON_TABLEAU is not None
    assert CROUZEIX_DIRK3_TABLEAU is not None
    assert GAUSS_LEGENDRE4_TABLEAU is not None
    assert LOBATTO_IIIC4_TABLEAU is not None
    assert RADAU_IIA5_TABLEAU is not None
    assert ARK324L2SA_TABLEAU is not None
    assert ARK436L2SA_TABLEAU is not None
    assert ARK437L2SA_TABLEAU is not None
    assert ARK548L2SA_TABLEAU is not None
    assert ARK548L2SAB_TABLEAU is not None
    assert SchemeKennedyCarpenter32 is not None
    assert SchemeKennedyCarpenter43_6 is not None
    assert SchemeKennedyCarpenter43_7 is not None
    assert SchemeKennedyCarpenter54 is not None
    assert SchemeKennedyCarpenter54b is not None
    assert SchemeBackwardEuler is not None
    assert SchemeImplicitMidpoint is not None
    assert SchemeCrankNicolson is not None
    assert SchemeCrouzeixDIRK3 is not None
    assert SchemeGaussLegendre4 is not None
    assert SchemeLobattoIIIC4 is not None
    assert SchemeRadauIIA5 is not None
    assert SchemeIMEXEuler is not None
    assert str(Integrator()) == "STARK integrator (safe mode)"
    assert repr(marcher) == "Marcher(scheme='MinimalScheme', executor=Executor(tolerance=Tolerance(atol=1e-08, rtol=1e-06), safety=Safety(progress=True, block_sizes=True, apply_delta=True), regulator=Regulator(safety=0.8, min_factor=0.1, max_factor=5.0, error_exponent=0.2), accelerator=AcceleratorAbsent(strict=False, values={}), values={}))"
    assert str(marcher) == "Marcher MinimalScheme with atol=1e-08, rtol=1e-06"
    assert "Auditor(status=" in repr(auditor)
    assert "ResolventPicard" in repr(auto_picard)
    assert "accelerator=AcceleratorAbsent" in repr(auto_picard)
    assert str(auto_picard) == "ResolventPicard"
    assert "ResolventCoupledPicard" in repr(auto_coupled_picard)
    assert str(auto_coupled_picard) == "ResolventCoupledPicard"
    assert "ResolventAnderson" in repr(auto_anderson)
    assert str(auto_anderson) == "ResolventAnderson"
    assert "ResolventBroyden" in repr(auto_broyden)
    assert str(auto_broyden) == "ResolventBroyden"
    assert "ResolventNewton" in repr(auto_newton)
    assert str(auto_newton) == "ResolventNewton"
    assert "ResolventCoupledNewton" in repr(auto_coupled_newton)
    assert str(auto_coupled_newton) == "ResolventCoupledNewton"


def test_scheme_classes_can_display_their_resolvent_problems() -> None:
    from stark.schemes.implicit_adaptive import SchemeKvaerno3
    from stark.schemes.imex_adaptive import SchemeKennedyCarpenter32

    implicit_text = SchemeKvaerno3.display_resolvent_problem()
    imex_text = SchemeKennedyCarpenter32.display_resolvent_problem()

    assert "Unknown stage block" in implicit_text
    assert "a_{ij}" in implicit_text
    assert "coupled block system" in implicit_text or "sequentially" in implicit_text
    assert "f_im" in imex_text
    assert "f_ex" in imex_text


def test_benchmark_packages_import() -> None:
    assert importlib.import_module("examples.comparison.brusselator_2d.common") is not None
    assert importlib.import_module("examples.comparison.fitzhugh_nagumo_1d.common") is not None
    assert importlib.import_module("examples.comparison.fput.common") is not None














