"""Catalogue of built-in method-stack components and curated stacks.

STARK methods are built as a stack:

    scheme -> resolvent -> inverter

Those layers are deliberately not interchangeable concepts. This catalogue
keeps their metadata separate, then names curated stack recipes for docs,
tests, comparison helpers, and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from stark.core.contracts import SchemeLike
from stark.methods.inverters import (
    InverterDense,
    InverterKrylovArnoldi,
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
)
from stark.methods.method import Method
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
from stark.methods.schemes import (
    SchemeBDF2,
    SchemeBackwardEuler,
    SchemeBogackiShampine,
    SchemeCashKarp,
    SchemeCrankNicolson,
    SchemeCrouzeixDIRK3,
    SchemeDormandPrince,
    SchemeEuler,
    SchemeFehlberg45,
    SchemeGaussLegendre4,
    SchemeHeun,
    SchemeIMEXEuler,
    SchemeImplicitMidpoint,
    SchemeKennedyCarpenter32,
    SchemeKennedyCarpenter43_6,
    SchemeKennedyCarpenter43_7,
    SchemeKennedyCarpenter54,
    SchemeKennedyCarpenter54b,
    SchemeKutta3,
    SchemeKvaerno3,
    SchemeKvaerno4,
    SchemeKvaerno5,
    SchemeLobattoIIIC4,
    SchemeMidpoint,
    SchemeRK4,
    SchemeRK38,
    SchemeRadauIIA5,
    SchemeRalston,
    SchemeSDIRK21,
    SchemeSSPRK33,
    SchemeTsitouras5,
)


class MethodCatalogueMaturity(str, Enum):
    """Release confidence for one catalogue component.

    These labels are deliberately conservative. Explicit schemes are marked
    stable because they have the smallest support stack and longest package
    history. Implicit, IMEX, linearized resolvent, and dense inverter paths are
    beta while release evidence is still being gathered. Known open-ended
    families such as Krylov, relaxation, fixed-point, and secant-style
    resolvents remain experimental until examples, safeguards, and benchmarks
    settle their public shape.
    """

    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"


class MethodCatalogueBenchmarkTier(str, Enum):
    """How aggressively benchmarks should cover a component or stack."""

    SMOKE = "smoke"
    REPRESENTATIVE = "representative"
    EXHAUSTIVE = "exhaustive"
    OMIT = "omit"


class MethodCatalogueSchemeFamily(str, Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    IMEX = "imex"


class MethodCatalogueSchemeStepping(str, Enum):
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class MethodCatalogueProblemClass(str, Enum):
    GENERAL = "general"
    NONSTIFF = "nonstiff"
    STIFF = "stiff"
    SPLIT = "split"


class MethodCatalogueResolventFamily(str, Enum):
    FIXED_POINT = "fixed-point"
    LINEARIZED = "linearized"
    SECANT = "secant"


class MethodCatalogueInverterFamily(str, Enum):
    DENSE = "dense"
    KRYLOV = "krylov"
    RELAXATION = "relaxation"


METHOD_CATALOGUE_MATURITY_RANK = {
    MethodCatalogueMaturity.EXPERIMENTAL: 0,
    MethodCatalogueMaturity.BETA: 1,
    MethodCatalogueMaturity.STABLE: 2,
}


@dataclass(frozen=True, slots=True)
class MethodCatalogueScheme:
    name: str
    scheme: type[SchemeLike]
    family: MethodCatalogueSchemeFamily
    stepping: MethodCatalogueSchemeStepping
    problem_class: MethodCatalogueProblemClass
    maturity: MethodCatalogueMaturity
    benchmark_tier: MethodCatalogueBenchmarkTier
    summary: str
    order: int | None = None
    embedded_order: int | None = None
    requires_resolvent: bool = False


@dataclass(frozen=True, slots=True)
class MethodCatalogueResolvent:
    name: str
    resolvent: type[Any]
    family: MethodCatalogueResolventFamily
    maturity: MethodCatalogueMaturity
    benchmark_tier: MethodCatalogueBenchmarkTier
    summary: str
    requires_linearizer: bool = False
    requires_inverter: bool = False
    supports_coupled_stages: bool = False


@dataclass(frozen=True, slots=True)
class MethodCatalogueInverter:
    name: str
    inverter: type[Any]
    family: MethodCatalogueInverterFamily
    maturity: MethodCatalogueMaturity
    benchmark_tier: MethodCatalogueBenchmarkTier
    summary: str
    matrix_free: bool = False
    supports_preconditioner: bool = False


@dataclass(frozen=True, slots=True)
class MethodCatalogueStack:
    """Curated method-stack recipe.

    Stack maturity is derived from the named components. A stack does not own
    independent maturity, because it is only as mature as its least mature
    scheme/resolvent/inverter member.
    """

    name: str
    scheme: str
    resolvent: str | None = None
    inverter: str | None = None
    problem_class: MethodCatalogueProblemClass = MethodCatalogueProblemClass.GENERAL
    benchmark_tier: MethodCatalogueBenchmarkTier = MethodCatalogueBenchmarkTier.OMIT
    summary: str = ""


@dataclass(frozen=True, slots=True)
class MethodCatalogueStackComponents:
    stack: MethodCatalogueStack
    scheme: MethodCatalogueScheme
    resolvent: MethodCatalogueResolvent | None
    inverter: MethodCatalogueInverter | None


@dataclass(frozen=True, slots=True)
class MethodCatalogue:
    schemes: tuple[MethodCatalogueScheme, ...]
    resolvents: tuple[MethodCatalogueResolvent, ...]
    inverters: tuple[MethodCatalogueInverter, ...]
    stacks: tuple[MethodCatalogueStack, ...]

    def scheme(self, name: str) -> MethodCatalogueScheme:
        return self.entry_by_name(self.schemes, name)

    def resolvent(self, name: str) -> MethodCatalogueResolvent:
        return self.entry_by_name(self.resolvents, name)

    def inverter(self, name: str) -> MethodCatalogueInverter:
        return self.entry_by_name(self.inverters, name)

    def stack(self, name: str) -> MethodCatalogueStack:
        return self.entry_by_name(self.stacks, name)

    def components(self, stack: str | MethodCatalogueStack) -> MethodCatalogueStackComponents:
        stack_entry = self.stack(stack) if isinstance(stack, str) else stack
        return MethodCatalogueStackComponents(
            stack=stack_entry,
            scheme=self.scheme(stack_entry.scheme),
            resolvent=(
                self.resolvent(stack_entry.resolvent)
                if stack_entry.resolvent is not None
                else None
            ),
            inverter=(
                self.inverter(stack_entry.inverter)
                if stack_entry.inverter is not None
                else None
            ),
        )

    def maturity(self, stack: str | MethodCatalogueStack) -> MethodCatalogueMaturity:
        components = self.components(stack)
        maturities = [components.scheme.maturity]
        if components.resolvent is not None:
            maturities.append(components.resolvent.maturity)
        if components.inverter is not None:
            maturities.append(components.inverter.maturity)
        return min(
            maturities,
            key=lambda maturity: METHOD_CATALOGUE_MATURITY_RANK[maturity],
        )

    def method(self, stack: str | MethodCatalogueStack) -> Method:
        components = self.components(stack)
        return Method(
            scheme=components.scheme.scheme,
            resolvent=(
                components.resolvent.resolvent
                if components.resolvent is not None
                else None
            ),
            inverter=(
                components.inverter.inverter
                if components.inverter is not None
                else None
            ),
        )

    def stacks_by_benchmark_tier(
        self,
        tier: MethodCatalogueBenchmarkTier,
    ) -> tuple[MethodCatalogueStack, ...]:
        return tuple(stack for stack in self.stacks if stack.benchmark_tier == tier)

    @staticmethod
    def entry_by_name(entries: tuple[Any, ...], name: str) -> Any:
        for entry in entries:
            if entry.name == name:
                return entry
        raise KeyError(name)


METHOD_CATALOGUE_SCHEMES = (
    MethodCatalogueScheme("SchemeEuler", SchemeEuler, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "First-order explicit Euler method.", order=1),
    MethodCatalogueScheme("SchemeMidpoint", SchemeMidpoint, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Second-order explicit midpoint method.", order=2),
    MethodCatalogueScheme("SchemeHeun", SchemeHeun, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Second-order explicit trapezoidal method.", order=2),
    MethodCatalogueScheme("SchemeRalston", SchemeRalston, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Second-order Ralston method.", order=2),
    MethodCatalogueScheme("SchemeKutta3", SchemeKutta3, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Third-order Kutta method.", order=3),
    MethodCatalogueScheme("SchemeSSPRK33", SchemeSSPRK33, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Third-order SSP Runge-Kutta method.", order=3),
    MethodCatalogueScheme("SchemeRK4", SchemeRK4, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.SMOKE, "Classical fourth-order Runge-Kutta method.", order=4),
    MethodCatalogueScheme("SchemeRK38", SchemeRK38, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Fourth-order 3/8-rule Runge-Kutta method.", order=4),
    MethodCatalogueScheme("SchemeBogackiShampine", SchemeBogackiShampine, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive explicit RK method for modest accuracy non-stiff solves.", order=3, embedded_order=2),
    MethodCatalogueScheme("SchemeCashKarp", SchemeCashKarp, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.SMOKE, "Adaptive explicit RK method for ordinary non-stiff solves.", order=5, embedded_order=4),
    MethodCatalogueScheme("SchemeDormandPrince", SchemeDormandPrince, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive Dormand-Prince method for non-stiff solves.", order=5, embedded_order=4),
    MethodCatalogueScheme("SchemeFehlberg45", SchemeFehlberg45, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Adaptive Fehlberg 4(5) method.", order=5, embedded_order=4),
    MethodCatalogueScheme("SchemeTsitouras5", SchemeTsitouras5, MethodCatalogueSchemeFamily.EXPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.NONSTIFF, MethodCatalogueMaturity.STABLE, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive Tsitouras fifth-order method.", order=5, embedded_order=4),
    MethodCatalogueScheme("SchemeBackwardEuler", SchemeBackwardEuler, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "First-order implicit method for stiff fixed-step solves.", order=1, requires_resolvent=True),
    MethodCatalogueScheme("SchemeImplicitMidpoint", SchemeImplicitMidpoint, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Second-order implicit midpoint method.", order=2, requires_resolvent=True),
    MethodCatalogueScheme("SchemeCrankNicolson", SchemeCrankNicolson, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Second-order Crank-Nicolson method.", order=2, requires_resolvent=True),
    MethodCatalogueScheme("SchemeCrouzeixDIRK3", SchemeCrouzeixDIRK3, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Third-order DIRK method.", order=3, requires_resolvent=True),
    MethodCatalogueScheme("SchemeGaussLegendre4", SchemeGaussLegendre4, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Fourth-order Gauss-Legendre collocation method.", order=4, requires_resolvent=True),
    MethodCatalogueScheme("SchemeLobattoIIIC4", SchemeLobattoIIIC4, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Fourth-order Lobatto IIIC method.", order=4, requires_resolvent=True),
    MethodCatalogueScheme("SchemeRadauIIA5", SchemeRadauIIA5, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Fifth-order Radau IIA method.", order=5, requires_resolvent=True),
    MethodCatalogueScheme("SchemeBDF2", SchemeBDF2, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive second-order BDF method.", order=2, embedded_order=1, requires_resolvent=True),
    MethodCatalogueScheme("SchemeSDIRK21", SchemeSDIRK21, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.OMIT, "Adaptive SDIRK 2(1) method; not a beta recommendation.", order=2, embedded_order=1, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKvaerno3", SchemeKvaerno3, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Adaptive Kvaerno 3(2) ESDIRK method.", order=3, embedded_order=2, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKvaerno4", SchemeKvaerno4, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive Kvaerno 4(3) ESDIRK method.", order=4, embedded_order=3, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKvaerno5", SchemeKvaerno5, MethodCatalogueSchemeFamily.IMPLICIT, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.STIFF, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.SMOKE, "Adaptive Kvaerno 5(4) ESDIRK method.", order=5, embedded_order=4, requires_resolvent=True),
    MethodCatalogueScheme("SchemeIMEXEuler", SchemeIMEXEuler, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.FIXED, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "First-order fixed-step IMEX Euler method.", order=1, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKennedyCarpenter32", SchemeKennedyCarpenter32, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive Kennedy-Carpenter IMEX 3(2) method.", order=3, embedded_order=2, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKennedyCarpenter43_6", SchemeKennedyCarpenter43_6, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Adaptive Kennedy-Carpenter IMEX 4(3) method with six stages.", order=4, embedded_order=3, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKennedyCarpenter43_7", SchemeKennedyCarpenter43_7, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Adaptive Kennedy-Carpenter IMEX 4(3) method with seven stages.", order=4, embedded_order=3, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKennedyCarpenter54", SchemeKennedyCarpenter54, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Adaptive Kennedy-Carpenter IMEX 5(4) method.", order=5, embedded_order=4, requires_resolvent=True),
    MethodCatalogueScheme("SchemeKennedyCarpenter54b", SchemeKennedyCarpenter54b, MethodCatalogueSchemeFamily.IMEX, MethodCatalogueSchemeStepping.ADAPTIVE, MethodCatalogueProblemClass.SPLIT, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Alternative adaptive Kennedy-Carpenter IMEX 5(4) method.", order=5, embedded_order=4, requires_resolvent=True),
)


METHOD_CATALOGUE_RESOLVENTS = (
    MethodCatalogueResolvent("ResolventPicard", ResolventPicard, MethodCatalogueResolventFamily.FIXED_POINT, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.OMIT, "Fixed-point resolvent for strongly contractive one-stage equations."),
    MethodCatalogueResolvent("ResolventCoupledPicard", ResolventCoupledPicard, MethodCatalogueResolventFamily.FIXED_POINT, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.OMIT, "Fixed-point resolvent for coupled stage equations.", supports_coupled_stages=True),
    MethodCatalogueResolvent("ResolventNewton", ResolventNewton, MethodCatalogueResolventFamily.LINEARIZED, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.SMOKE, "Newton resolvent for one-stage implicit equations.", requires_linearizer=True, requires_inverter=True),
    MethodCatalogueResolvent("ResolventCoupledNewton", ResolventCoupledNewton, MethodCatalogueResolventFamily.LINEARIZED, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Newton resolvent for coupled implicit stage systems.", requires_linearizer=True, requires_inverter=True, supports_coupled_stages=True),
    MethodCatalogueResolvent("ResolventChord", ResolventChord, MethodCatalogueResolventFamily.LINEARIZED, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.REPRESENTATIVE, "Chord resolvent that reuses a linearization within a solve.", requires_linearizer=True, requires_inverter=True),
    MethodCatalogueResolvent("ResolventVeryChord", ResolventVeryChord, MethodCatalogueResolventFamily.LINEARIZED, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "More aggressive chord-style resolvent with broader reuse.", requires_linearizer=True, requires_inverter=True),
    MethodCatalogueResolvent("ResolventAnderson", ResolventAnderson, MethodCatalogueResolventFamily.SECANT, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.OMIT, "Anderson-accelerated fixed-point resolvent."),
    MethodCatalogueResolvent("ResolventBroyden", ResolventBroyden, MethodCatalogueResolventFamily.SECANT, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.OMIT, "Broyden-style secant resolvent."),
)


METHOD_CATALOGUE_INVERTERS = (
    MethodCatalogueInverter("InverterDense", InverterDense, MethodCatalogueInverterFamily.DENSE, MethodCatalogueMaturity.BETA, MethodCatalogueBenchmarkTier.SMOKE, "Small-system dense inverse-action default."),
    MethodCatalogueInverter("InverterKrylovArnoldi", InverterKrylovArnoldi, MethodCatalogueInverterFamily.KRYLOV, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Matrix-free Arnoldi/Krylov inverse-action path.", matrix_free=True, supports_preconditioner=True),
    MethodCatalogueInverter("InverterRelaxationJacobi", InverterRelaxationJacobi, MethodCatalogueInverterFamily.RELAXATION, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Jacobi relaxation inverse-action path.", matrix_free=True),
    MethodCatalogueInverter("InverterRelaxationRichardson", InverterRelaxationRichardson, MethodCatalogueInverterFamily.RELAXATION, MethodCatalogueMaturity.EXPERIMENTAL, MethodCatalogueBenchmarkTier.EXHAUSTIVE, "Richardson relaxation inverse-action path.", matrix_free=True),
)


METHOD_CATALOGUE_STACKS = (
    MethodCatalogueStack(name="euler", scheme="SchemeEuler", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Forward Euler explicit stack."),
    MethodCatalogueStack(name="midpoint", scheme="SchemeMidpoint", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Explicit midpoint stack."),
    MethodCatalogueStack(name="heun", scheme="SchemeHeun", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Heun explicit stack."),
    MethodCatalogueStack(name="ralston", scheme="SchemeRalston", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Ralston explicit stack."),
    MethodCatalogueStack(name="kutta3", scheme="SchemeKutta3", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kutta third-order explicit stack."),
    MethodCatalogueStack(name="ssprk33", scheme="SchemeSSPRK33", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="SSP RK33 explicit stack."),
    MethodCatalogueStack(name="rk4", scheme="SchemeRK4", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.SMOKE, summary="Classical RK4 explicit stack."),
    MethodCatalogueStack(name="rk38", scheme="SchemeRK38", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="3/8-rule RK4 explicit stack."),
    MethodCatalogueStack(name="bs23", scheme="SchemeBogackiShampine", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Bogacki-Shampine adaptive explicit stack."),
    MethodCatalogueStack(name="rkck", scheme="SchemeCashKarp", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.SMOKE, summary="Cash-Karp adaptive explicit stack."),
    MethodCatalogueStack(name="rkdp", scheme="SchemeDormandPrince", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Dormand-Prince adaptive explicit stack."),
    MethodCatalogueStack(name="rkf45", scheme="SchemeFehlberg45", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Fehlberg 4(5) adaptive explicit stack."),
    MethodCatalogueStack(name="tsit5", scheme="SchemeTsitouras5", problem_class=MethodCatalogueProblemClass.NONSTIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Tsitouras 5 adaptive explicit stack."),
    MethodCatalogueStack(name="be-newton-dense", scheme="SchemeBackwardEuler", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Backward Euler with Newton and dense inversion."),
    MethodCatalogueStack(name="be-chord-dense", scheme="SchemeBackwardEuler", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Backward Euler with chord iteration and dense inversion."),
    MethodCatalogueStack(name="be-verychord-dense", scheme="SchemeBackwardEuler", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Backward Euler with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="be-picard", scheme="SchemeBackwardEuler", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Backward Euler with Picard iteration."),
    MethodCatalogueStack(name="im-newton-dense", scheme="SchemeImplicitMidpoint", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Implicit midpoint with Newton and dense inversion."),
    MethodCatalogueStack(name="im-chord-dense", scheme="SchemeImplicitMidpoint", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Implicit midpoint with chord iteration and dense inversion."),
    MethodCatalogueStack(name="im-verychord-dense", scheme="SchemeImplicitMidpoint", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Implicit midpoint with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="im-picard", scheme="SchemeImplicitMidpoint", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Implicit midpoint with Picard iteration."),
    MethodCatalogueStack(name="cn-newton-dense", scheme="SchemeCrankNicolson", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Crank-Nicolson with Newton and dense inversion."),
    MethodCatalogueStack(name="cn-chord-dense", scheme="SchemeCrankNicolson", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Crank-Nicolson with chord iteration and dense inversion."),
    MethodCatalogueStack(name="cn-verychord-dense", scheme="SchemeCrankNicolson", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Crank-Nicolson with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="cn-picard", scheme="SchemeCrankNicolson", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Crank-Nicolson with Picard iteration."),
    MethodCatalogueStack(name="crouzeix3-newton-dense", scheme="SchemeCrouzeixDIRK3", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Crouzeix DIRK3 with Newton and dense inversion."),
    MethodCatalogueStack(name="crouzeix3-chord-dense", scheme="SchemeCrouzeixDIRK3", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Crouzeix DIRK3 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="crouzeix3-verychord-dense", scheme="SchemeCrouzeixDIRK3", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Crouzeix DIRK3 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="crouzeix3-picard", scheme="SchemeCrouzeixDIRK3", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Crouzeix DIRK3 with Picard iteration."),
    MethodCatalogueStack(name="gl4-coupled-newton-dense", scheme="SchemeGaussLegendre4", resolvent="ResolventCoupledNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Gauss-Legendre 4 with coupled Newton and dense inversion."),
    MethodCatalogueStack(name="gl4-coupled-picard", scheme="SchemeGaussLegendre4", resolvent="ResolventCoupledPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Gauss-Legendre 4 with coupled Picard iteration."),
    MethodCatalogueStack(name="lobatto4-coupled-newton-dense", scheme="SchemeLobattoIIIC4", resolvent="ResolventCoupledNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Lobatto IIIC 4 with coupled Newton and dense inversion."),
    MethodCatalogueStack(name="lobatto4-coupled-picard", scheme="SchemeLobattoIIIC4", resolvent="ResolventCoupledPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Lobatto IIIC 4 with coupled Picard iteration."),
    MethodCatalogueStack(name="radau5-coupled-newton-dense", scheme="SchemeRadauIIA5", resolvent="ResolventCoupledNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Radau IIA 5 with coupled Newton and dense inversion."),
    MethodCatalogueStack(name="radau5-coupled-picard", scheme="SchemeRadauIIA5", resolvent="ResolventCoupledPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Radau IIA 5 with coupled Picard iteration."),
    MethodCatalogueStack(name="bdf2-newton-dense", scheme="SchemeBDF2", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="BDF2 with Newton and dense inversion."),
    MethodCatalogueStack(name="bdf2-chord-dense", scheme="SchemeBDF2", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="BDF2 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="bdf2-verychord-dense", scheme="SchemeBDF2", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="BDF2 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="bdf2-picard", scheme="SchemeBDF2", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="BDF2 with Picard iteration."),
    MethodCatalogueStack(name="sdirk21-newton-dense", scheme="SchemeSDIRK21", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="SDIRK21 with Newton and dense inversion."),
    MethodCatalogueStack(name="sdirk21-chord-dense", scheme="SchemeSDIRK21", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="SDIRK21 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="sdirk21-verychord-dense", scheme="SchemeSDIRK21", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="SDIRK21 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="sdirk21-picard", scheme="SchemeSDIRK21", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="SDIRK21 with Picard iteration."),
    MethodCatalogueStack(name="kvaerno3-newton-dense", scheme="SchemeKvaerno3", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kvaerno3 with Newton and dense inversion."),
    MethodCatalogueStack(name="kvaerno3-chord-dense", scheme="SchemeKvaerno3", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kvaerno3 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno3-verychord-dense", scheme="SchemeKvaerno3", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kvaerno3 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno3-picard", scheme="SchemeKvaerno3", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kvaerno3 with Picard iteration."),
    MethodCatalogueStack(name="kvaerno4-newton-dense", scheme="SchemeKvaerno4", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Kvaerno4 with Newton and dense inversion."),
    MethodCatalogueStack(name="kvaerno4-chord-dense", scheme="SchemeKvaerno4", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Kvaerno4 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno4-verychord-dense", scheme="SchemeKvaerno4", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kvaerno4 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno4-picard", scheme="SchemeKvaerno4", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kvaerno4 with Picard iteration."),
    MethodCatalogueStack(name="kvaerno5-newton-dense", scheme="SchemeKvaerno5", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.SMOKE, summary="Kvaerno5 with Newton and dense inversion."),
    MethodCatalogueStack(name="kvaerno5-chord-dense", scheme="SchemeKvaerno5", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Kvaerno5 with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno5-verychord-dense", scheme="SchemeKvaerno5", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kvaerno5 with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kvaerno5-picard", scheme="SchemeKvaerno5", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.STIFF, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kvaerno5 with Picard iteration."),
    MethodCatalogueStack(name="imexeuler-newton-dense", scheme="SchemeIMEXEuler", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="IMEX Euler with Newton and dense inversion."),
    MethodCatalogueStack(name="imexeuler-chord-dense", scheme="SchemeIMEXEuler", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="IMEX Euler with chord iteration and dense inversion."),
    MethodCatalogueStack(name="imexeuler-verychord-dense", scheme="SchemeIMEXEuler", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="IMEX Euler with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="imexeuler-picard", scheme="SchemeIMEXEuler", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="IMEX Euler with Picard iteration."),
    MethodCatalogueStack(name="kc32-newton-dense", scheme="SchemeKennedyCarpenter32", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Kennedy-Carpenter 3(2) with Newton and dense inversion."),
    MethodCatalogueStack(name="kc32-chord-dense", scheme="SchemeKennedyCarpenter32", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 3(2) with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc32-verychord-dense", scheme="SchemeKennedyCarpenter32", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 3(2) with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc32-picard", scheme="SchemeKennedyCarpenter32", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kennedy-Carpenter 3(2) with Picard iteration."),
    MethodCatalogueStack(name="kc43-6-newton-dense", scheme="SchemeKennedyCarpenter43_6", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 6-stage with Newton and dense inversion."),
    MethodCatalogueStack(name="kc43-6-chord-dense", scheme="SchemeKennedyCarpenter43_6", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 6-stage with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc43-6-verychord-dense", scheme="SchemeKennedyCarpenter43_6", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 6-stage with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc43-6-picard", scheme="SchemeKennedyCarpenter43_6", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kennedy-Carpenter 4(3) 6-stage with Picard iteration."),
    MethodCatalogueStack(name="kc43-7-newton-dense", scheme="SchemeKennedyCarpenter43_7", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 7-stage with Newton and dense inversion."),
    MethodCatalogueStack(name="kc43-7-chord-dense", scheme="SchemeKennedyCarpenter43_7", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 7-stage with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc43-7-verychord-dense", scheme="SchemeKennedyCarpenter43_7", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 4(3) 7-stage with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc43-7-picard", scheme="SchemeKennedyCarpenter43_7", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kennedy-Carpenter 4(3) 7-stage with Picard iteration."),
    MethodCatalogueStack(name="kc54-newton-dense", scheme="SchemeKennedyCarpenter54", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.REPRESENTATIVE, summary="Kennedy-Carpenter 5(4) with Newton and dense inversion."),
    MethodCatalogueStack(name="kc54-chord-dense", scheme="SchemeKennedyCarpenter54", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 5(4) with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc54-verychord-dense", scheme="SchemeKennedyCarpenter54", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 5(4) with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc54-picard", scheme="SchemeKennedyCarpenter54", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kennedy-Carpenter 5(4) with Picard iteration."),
    MethodCatalogueStack(name="kc54b-newton-dense", scheme="SchemeKennedyCarpenter54b", resolvent="ResolventNewton", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 5(4)b with Newton and dense inversion."),
    MethodCatalogueStack(name="kc54b-chord-dense", scheme="SchemeKennedyCarpenter54b", resolvent="ResolventChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 5(4)b with chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc54b-verychord-dense", scheme="SchemeKennedyCarpenter54b", resolvent="ResolventVeryChord", inverter="InverterDense", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.EXHAUSTIVE, summary="Kennedy-Carpenter 5(4)b with very-chord iteration and dense inversion."),
    MethodCatalogueStack(name="kc54b-picard", scheme="SchemeKennedyCarpenter54b", resolvent="ResolventPicard", problem_class=MethodCatalogueProblemClass.SPLIT, benchmark_tier=MethodCatalogueBenchmarkTier.OMIT, summary="Kennedy-Carpenter 5(4)b with Picard iteration."),
)


METHOD_CATALOGUE = MethodCatalogue(
    schemes=METHOD_CATALOGUE_SCHEMES,
    resolvents=METHOD_CATALOGUE_RESOLVENTS,
    inverters=METHOD_CATALOGUE_INVERTERS,
    stacks=METHOD_CATALOGUE_STACKS,
)


__all__ = [
    "MethodCatalogue",
    "MethodCatalogueBenchmarkTier",
    "MethodCatalogueInverter",
    "MethodCatalogueInverterFamily",
    "METHOD_CATALOGUE_INVERTERS",
    "METHOD_CATALOGUE",
    "MethodCatalogueMaturity",
    "METHOD_CATALOGUE_MATURITY_RANK",
    "MethodCatalogueProblemClass",
    "MethodCatalogueResolvent",
    "MethodCatalogueResolventFamily",
    "METHOD_CATALOGUE_RESOLVENTS",
    "MethodCatalogueScheme",
    "MethodCatalogueSchemeFamily",
    "MethodCatalogueSchemeStepping",
    "METHOD_CATALOGUE_SCHEMES",
    "MethodCatalogueStack",
    "MethodCatalogueStackComponents",
    "METHOD_CATALOGUE_STACKS",
]
