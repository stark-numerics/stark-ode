from __future__ import annotations

from dataclasses import dataclass
from inspect import signature

import pytest

import stark.schemes as schemes


GENERATED = "generated"
HOOK_ONLY = "hook-only"


@dataclass(frozen=True, slots=True)
class AlgebraistCapability:
    scheme_cls: type
    family: str
    capability: str
    reason: str | None = None

    @property
    def name(self) -> str:
        return self.scheme_cls.__name__


ALGEBRAIST_CAPABILITIES = (
    AlgebraistCapability(schemes.SchemeEuler, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeHeun, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeMidpoint, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeRalston, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeKutta3, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeSSPRK33, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeRK4, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeRK38, "explicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeBogackiShampine, "explicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeCashKarp, "explicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeDormandPrince, "explicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeFehlberg45, "explicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeTsitouras5, "explicit adaptive", GENERATED),
    AlgebraistCapability(
        schemes.SchemeBackwardEuler,
        "implicit fixed",
        HOOK_ONLY,
        "one-stage formula has no useful generated stage algebra",
    ),
    AlgebraistCapability(schemes.SchemeCrankNicolson, "implicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeImplicitMidpoint, "implicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeCrouzeixDIRK3, "implicit fixed", GENERATED),
    AlgebraistCapability(schemes.SchemeGaussLegendre4, "implicit fixed", GENERATED),
    AlgebraistCapability(
        schemes.SchemeLobattoIIIC4,
        "implicit fixed",
        HOOK_ONLY,
        "dense coupled stages remain on the generic resolvent path",
    ),
    AlgebraistCapability(
        schemes.SchemeRadauIIA5,
        "implicit fixed",
        HOOK_ONLY,
        "dense coupled stages remain on the generic resolvent path",
    ),
    AlgebraistCapability(
        schemes.SchemeBDF2,
        "implicit adaptive",
        HOOK_ONLY,
        "multistep update is not algebra-heavy enough to generate",
    ),
    AlgebraistCapability(schemes.SchemeSDIRK21, "implicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKvaerno3, "implicit adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKvaerno4, "implicit adaptive", GENERATED),
    AlgebraistCapability(
        schemes.SchemeIMEXEuler,
        "IMEX fixed",
        HOOK_ONLY,
        "one-stage IMEX formula is not algebra-heavy enough to generate",
    ),
    AlgebraistCapability(schemes.SchemeKennedyCarpenter32, "IMEX adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKennedyCarpenter43_6, "IMEX adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKennedyCarpenter43_7, "IMEX adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKennedyCarpenter54, "IMEX adaptive", GENERATED),
    AlgebraistCapability(schemes.SchemeKennedyCarpenter54b, "IMEX adaptive", GENERATED),
)


def capability_ids(capability: AlgebraistCapability) -> str:
    return f"{capability.family}: {capability.name} -> {capability.capability}"


def public_builtin_scheme_names() -> set[str]:
    names: set[str] = set()
    for name in schemes.__all__:
        exported = getattr(schemes, name)
        if not isinstance(exported, type):
            continue
        if "algebraist" not in signature(exported).parameters:
            continue
        names.add(name)
    return names


def test_algebraist_capability_matrix_enumerates_every_builtin_scheme() -> None:
    matrix_names = {capability.name for capability in ALGEBRAIST_CAPABILITIES}

    assert matrix_names == public_builtin_scheme_names()


def test_algebraist_capability_matrix_classifies_each_scheme_once() -> None:
    names = [capability.name for capability in ALGEBRAIST_CAPABILITIES]

    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "capability",
    ALGEBRAIST_CAPABILITIES,
    ids=capability_ids,
)
def test_algebraist_capability_matches_public_scheme_surface(
    capability: AlgebraistCapability,
) -> None:
    scheme_cls = capability.scheme_cls
    parameters = signature(scheme_cls).parameters

    assert "algebraist" in parameters

    if capability.capability == GENERATED:
        assert capability.reason is None
        assert "bind_algebraist_path" in scheme_cls.__dict__
        assert "call_algebraist" in scheme_cls.__dict__
        return

    assert capability.capability == HOOK_ONLY
    assert capability.reason
    assert "bind_algebraist_path" not in scheme_cls.__dict__
    assert "call_algebraist" not in scheme_cls.__dict__


def test_algebraist_capability_matrix_names_supported_families() -> None:
    families = {capability.family for capability in ALGEBRAIST_CAPABILITIES}

    assert families == {
        "explicit fixed",
        "explicit adaptive",
        "implicit fixed",
        "implicit adaptive",
        "IMEX fixed",
        "IMEX adaptive",
    }
