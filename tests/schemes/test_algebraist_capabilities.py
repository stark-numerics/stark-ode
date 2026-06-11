from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import pytest

import stark.methods.schemes as schemes

SPECIALIST = "specialist"
HOOK_ONLY = "hook-only"


@dataclass(frozen=True, slots=True)
class SchemeCapability:
    scheme_cls: type
    family: str
    capability: str
    reason: str | None = None

    @property
    def name(self) -> str:
        return self.scheme_cls.__name__


SCHEME_CAPABILITIES = (
    SchemeCapability(schemes.SchemeEuler, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeHeun, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeMidpoint, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeRalston, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeKutta3, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeSSPRK33, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeRK4, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeRK38, "explicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeBogackiShampine, "explicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeCashKarp, "explicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeDormandPrince, "explicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeFehlberg45, "explicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeTsitouras5, "explicit adaptive", SPECIALIST),
    SchemeCapability(
        schemes.SchemeBackwardEuler,
        "implicit fixed",
        HOOK_ONLY,
        "one-stage formula has no useful generated stage algebra",
    ),
    SchemeCapability(schemes.SchemeCrankNicolson, "implicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeImplicitMidpoint, "implicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeCrouzeixDIRK3, "implicit fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeGaussLegendre4, "implicit fixed", SPECIALIST),
    SchemeCapability(
        schemes.SchemeLobattoIIIC4,
        "implicit fixed",
        HOOK_ONLY,
        "dense coupled stages remain on the generic resolvent path",
    ),
    SchemeCapability(
        schemes.SchemeRadauIIA5,
        "implicit fixed",
        HOOK_ONLY,
        "dense coupled stages remain on the generic resolvent path",
    ),
    SchemeCapability(
        schemes.SchemeBDF2,
        "implicit adaptive",
        HOOK_ONLY,
        "multistep update is not algebra-heavy enough to generate",
    ),
    SchemeCapability(schemes.SchemeSDIRK21, "implicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKvaerno3, "implicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKvaerno4, "implicit adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeIMEXEuler, "IMEX fixed", SPECIALIST),
    SchemeCapability(schemes.SchemeKennedyCarpenter32, "IMEX adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKennedyCarpenter43_6, "IMEX adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKennedyCarpenter43_7, "IMEX adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKennedyCarpenter54, "IMEX adaptive", SPECIALIST),
    SchemeCapability(schemes.SchemeKennedyCarpenter54b, "IMEX adaptive", SPECIALIST),
)


def capability_ids(capability: SchemeCapability) -> str:
    return f"{capability.family}: {capability.name} -> {capability.capability}"


def public_builtin_scheme_names() -> set[str]:
    names: set[str] = set()

    for name in schemes.__all__:
        exported = getattr(schemes, name)
        if (
            isinstance(exported, type)
            and name.startswith("Scheme")
            and name != "SchemeDescriptor"
        ):
            names.add(name)

    return names


def test_scheme_capability_matrix_enumerates_every_builtin_scheme() -> None:
    matrix_names = {capability.name for capability in SCHEME_CAPABILITIES}

    assert matrix_names == public_builtin_scheme_names()


def test_scheme_capability_matrix_classifies_each_scheme_once() -> None:
    names = [capability.name for capability in SCHEME_CAPABILITIES]

    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "capability",
    SCHEME_CAPABILITIES,
    ids=capability_ids,
)
def test_scheme_capability_matches_public_scheme_surface(
    capability: SchemeCapability,
) -> None:
    scheme_cls = capability.scheme_cls
    parameters = signature(scheme_cls).parameters

    if capability.capability == SPECIALIST:
        assert capability.reason is None
        assert "specialist" in parameters
        assert "algebraist" not in parameters
        assert hasattr(scheme_cls, "prepare_specialized_kernels")
        assert hasattr(scheme_cls, "call_specialized")
        return

    assert capability.capability == HOOK_ONLY
    assert capability.reason is not None
    assert "specialist" in parameters
    assert "algebraist" not in parameters


def test_scheme_capability_matrix_names_supported_families() -> None:
    families = {capability.family for capability in SCHEME_CAPABILITIES}

    assert families == {
        "explicit fixed",
        "explicit adaptive",
        "implicit fixed",
        "implicit adaptive",
        "IMEX fixed",
        "IMEX adaptive",
    }
