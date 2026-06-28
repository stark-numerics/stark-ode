"""Basic smoke tests for the project scaffold."""

import importlib

from stark import Configuration, Interval, Tolerance
from stark.core import Auditor, Integrator, IntegratorStepper
from stark.engines import Accelerator, AcceleratorNone
from stark.core.block.operator import BlockOperatorDiagonal
from stark.diagnostics.comparison import ComparisonRunner, ComparisonEntryStepper, ComparisonProblemManual
from stark.core.block import Block
from stark.core.contracts import Resolvent
from stark.methods.inverters.support import InverterDescriptor
from stark.methods.resolvents import (
    ResolventAnderson,
    ResolventBroyden,
    ResolventChord,
    ResolventCoupledNewton,
    ResolventCoupledPicard,
    ResolventNewton,
    ResolventPicard,
    ResolventVeryChord,
)
from stark.methods.resolvents.method.descriptor import ResolventDescriptor
from stark.methods.schemes import (
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
from stark.methods.schemes.imex.adaptive import (
    ARK324L2SA_TABLEAU,
    ARK436L2SA_TABLEAU,
    ARK437L2SA_TABLEAU,
    ARK548L2SAB_TABLEAU,
    ARK548L2SA_TABLEAU,
)
from stark.methods.schemes.imex.fixed import IMEX_EULER_TABLEAU
from stark.methods.schemes.implicit.fixed import (
    BE_TABLEAU,
    CRANK_NICOLSON_TABLEAU,
    CROUZEIX_DIRK3_TABLEAU,
    GAUSS_LEGENDRE4_TABLEAU,
    IMPLICIT_MIDPOINT_TABLEAU,
    LOBATTO_IIIC4_TABLEAU,
    RADAU_IIA5_TABLEAU,
)
from stark.methods.schemes.method.tableau import ButcherTableau, ButcherTableauImex


def test_package_imports() -> None:
    """The top-level package should import cleanly."""
    assert importlib.import_module("stark") is not None
    assert Accelerator is not None
    assert Resolvent is not None


def test_comparator_module_imports() -> None:
    """The ComparisonRunner module should exist and import cleanly."""
    assert importlib.import_module("stark.diagnostics.comparison") is not None


def test_stepper_module_imports() -> None:
    """The stepper module should exist and import cleanly."""
    assert importlib.import_module("stark.core.integrator.stepper") is not None


def test_audit_module_imports() -> None:
    """The audit module should exist and import cleanly."""
    assert importlib.import_module("stark.core.auditor") is not None


def test_Configuration_module_imports() -> None:
    """Configuration lives in core with narrow domain protocol views."""
    assert importlib.import_module("stark.engines.shared.accelerators") is not None
    assert importlib.import_module("stark.core.configuration") is not None
    assert importlib.import_module("stark.core.tolerance") is not None
    assert importlib.import_module("stark.methods.schemes.configuration") is not None
    assert importlib.import_module("stark.methods.resolvents.configuration") is not None
    assert importlib.import_module("stark.methods.inverters.configuration") is not None
    assert importlib.import_module("stark.methods.schemes.method.tableau") is not None
    assert importlib.import_module("stark.methods.schemes.execution.step_support") is not None
    assert importlib.import_module("stark.engines.shared.algebraist.runtime") is not None
    assert importlib.import_module("stark.methods.schemes.display.display") is not None


def test_integrate_module_imports() -> None:
    """The integrate module should exist and import cleanly."""
    assert importlib.import_module("stark.core.integrator.integrator") is not None


def test_inverter_imports() -> None:
    """The inverter package should import cleanly."""
    inverters = importlib.import_module("stark.methods.inverters")

    assert inverters.InverterRelaxationRichardson is not None
    assert inverters.InverterRelaxationJacobi is not None


def test_resolvent_imports() -> None:
    """The resolvent package should expose methods and metadata cleanly."""
    resolvents = importlib.import_module("stark.methods.resolvents")
    equations = importlib.import_module("stark.methods.resolvents.equations")

    assert resolvents.ResolventConfiguration is not None
    assert resolvents.ResolventError is not None
    assert resolvents.ResolventPicard is not None
    assert resolvents.ResolventChord is not None
    assert resolvents.ResolventNewton is not None
    assert resolvents.ResolventVeryChord is not None
    assert resolvents.ResolventAnderson is not None
    assert resolvents.ResolventBroyden is not None
    assert resolvents.ResolventCoupledPicard is not None
    assert resolvents.ResolventCoupledNewton is not None
    assert equations.ResolventImplicitEquation is not None


def test_scheme_imports() -> None:
    """The schemes package should expose aggregate and grouped public imports."""
    schemes = importlib.import_module("stark.methods.schemes")
    adaptive = importlib.import_module("stark.methods.schemes.explicit.adaptive")
    adaptive_implicit = importlib.import_module("stark.methods.schemes.implicit.adaptive")
    fixed_step = importlib.import_module("stark.methods.schemes.explicit.fixed")
    imex_adaptive = importlib.import_module("stark.methods.schemes.imex.adaptive")
    imex_fixed = importlib.import_module("stark.methods.schemes.imex.fixed")
    implicit = importlib.import_module("stark.methods.schemes.implicit.fixed")

    assert schemes.SchemeDormandPrince is adaptive.SchemeDormandPrince
    assert schemes.SchemeCashKarp is adaptive.SchemeCashKarp
    assert schemes.SchemeTsitouras5 is adaptive.SchemeTsitouras5
    assert schemes.SchemeBDF2 is adaptive_implicit.SchemeBDF2
    assert schemes.SchemeKvaerno3 is adaptive_implicit.SchemeKvaerno3
    assert schemes.SchemeKvaerno4 is adaptive_implicit.SchemeKvaerno4
    assert schemes.SchemeKvaerno5 is adaptive_implicit.SchemeKvaerno5
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
    assert not hasattr(schemes, "GAUSS_LEGENDRE4_TABLEAU")
    assert not hasattr(schemes, "IMEX_EULER_TABLEAU")


class MinimalScheme:
    def __call__(self, interval: Interval, state: object) -> float:
        del interval, state
        return 0.0

    def snapshot_state(self, state: object) -> object:
        return state


class MinimalAllocator:
    def allocate_state(self) -> object:
        return object()

    def copy_state(self, source: object, out: object) -> None:
        del out, source

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
    def scale(a: float, x: "MinimalTranslation", out: "MinimalTranslation") -> "MinimalTranslation":
        del a, x
        return out

    @staticmethod
    def combine2(
        a0: float,
        x0: "MinimalTranslation",
        a1: float,
        x1: "MinimalTranslation",
        out: "MinimalTranslation",
    ) -> "MinimalTranslation":
        del a0, x0, a1, x1
        return out

    linear_combine = [scale, combine2]


class MinimalInverter:
    def bind(self, operator: object) -> None:
        del operator

    def __call__(self, rhs: Block, out: Block) -> None:
        del rhs, out


def test_core_objects_have_readable_representations() -> None:
    bakeoff_problem = ComparisonProblemManual(
        "Dummy",
        build_state=lambda: object(),
        build_interval=lambda: object(),
        difference=lambda left, right: 0.0,
    )
    bakeoff_entry = ComparisonEntryStepper("Dummy", lambda: object())
    bakeoff = ComparisonRunner(
        bakeoff_problem,
        [bakeoff_entry, ComparisonEntryStepper("Other", lambda: object())],
        repeats=1,
    )
    interval = Interval(0.0, 0.1, 1.0)
    block = Block([])
    block_operator = BlockOperatorDiagonal([])
    inverter_budget = Configuration()
    inverter_tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    inverter_descriptor = InverterDescriptor("Richardson", "Richardson relaxation")
    resolver_policy = Configuration()
    resolver_tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    resolver_descriptor = ResolventDescriptor("Picard", "Picard Iteration")
    configuration = Configuration()
    configuration_tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    scheme_tolerance = Tolerance(atol=1.0e-8, rtol=1.0e-6)
    tableau = ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E")
    imex_tableau = ButcherTableauImex(
        explicit=ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="E"),
        implicit=ButcherTableau(c=(0.0,), a=((),), b=(1.0,), order=1, short_name="I"),
    )
    stepper = IntegratorStepper(MinimalScheme())
    auditor = Auditor(interval=interval, stepper=stepper, snapshots=True, exercise=False)
    allocator = MinimalAllocator()
    auto_picard = ResolventPicard(allocator, accelerator=AcceleratorNone())
    auto_coupled_picard = ResolventCoupledPicard(
        allocator,
        tableau=ButcherTableau(
            c=(0.5, 0.5),
            a=((0.25, 0.25), (0.25, 0.25)),
            b=(0.5, 0.5),
            order=2,
            short_name="DummyCoupled",
        ),
    )
    auto_anderson = ResolventAnderson(
        allocator,
        inner_product=lambda left, right: 0.0,
    )
    auto_broyden = ResolventBroyden(
        allocator,
        inner_product=lambda left, right: 0.0,
    )
    auto_newton = ResolventNewton(
        allocator,
        linearizer=lambda interval, state, out: setattr(out, "apply", lambda translation, result: None),
        inverter=MinimalInverter(),
    )
    auto_coupled_newton = ResolventCoupledNewton(
        allocator,
        tableau=ButcherTableau(
            c=(0.5, 0.5),
            a=((0.25, 0.25), (0.25, 0.25)),
            b=(0.5, 0.5),
            order=2,
            short_name="DummyCoupled",
        ),
        linearizer=lambda interval, state, out: setattr(out, "apply", lambda translation, result: None),
        inverter=MinimalInverter(),
    )

    assert repr(interval) == "Interval(present=0.0, step=0.1, stop=1.0)"
    assert repr(bakeoff_problem).startswith("ComparisonProblemManual(")
    assert repr(bakeoff_entry).startswith("ComparisonEntryStepper(")
    assert "ComparisonRunner(" in repr(bakeoff)
    assert str(interval) == "[0, 1] step=0.1"
    assert repr(block) == "Block(size=0)"
    assert str(block) == "block[0]"
    assert repr(block_operator).startswith("BlockOperatorDiagonal(")
    assert str(block_operator) == "block operator[0]"
    assert inverter_budget.inverter_maximum_steps == 32
    assert repr(inverter_tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert repr(inverter_descriptor) == "InverterDescriptor(short_name='Richardson', full_name='Richardson relaxation')"
    assert resolver_policy.resolvent_maximum_steps == 16
    assert repr(resolver_tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert repr(resolver_descriptor) == "ResolventDescriptor(short_name='Picard', full_name='Picard Iteration')"
    assert configuration.check_progress is False
    assert repr(configuration_tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert repr(scheme_tolerance) == "Tolerance(atol=1e-08, rtol=1e-06)"
    assert str(configuration_tolerance) == "atol=1e-08, rtol=1e-06"
    assert repr(tableau) == "ButcherTableau(stages=1, order=1, embedded_order=None, name='E')"
    assert "ButcherTableauImex" in repr(imex_tableau)
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
    assert str(Integrator()) == "STARK integrator (fast mode)"
    assert repr(stepper) == "IntegratorStepper(scheme='MinimalScheme')"
    assert str(stepper) == "IntegratorStepper MinimalScheme"
    assert "Auditor(status=" in repr(auditor)
    assert "ResolventPicard" in repr(auto_picard)
    assert "accelerator=AcceleratorNone" in repr(auto_picard)
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
    from stark.methods.schemes.implicit.adaptive import SchemeKvaerno3
    from stark.methods.schemes.imex.adaptive import SchemeKennedyCarpenter32

    implicit_text = SchemeKvaerno3.display_resolvent_problem()
    imex_text = SchemeKennedyCarpenter32.display_resolvent_problem()

    assert "Unknown stage block" in implicit_text
    assert "a_{ij}" in implicit_text
    assert "coupled block system" in implicit_text or "sequentially" in implicit_text
    assert "f_im" in imex_text
    assert "f_ex" in imex_text


def test_benchmark_packages_import() -> None:
    assert importlib.import_module("competition.brusselator_2d.common") is not None
    assert importlib.import_module("competition.fitzhugh_nagumo_1d.common") is not None
    assert importlib.import_module("competition.fput.common") is not None
