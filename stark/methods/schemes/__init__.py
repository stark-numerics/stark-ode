"""Built-in scheme catalogue.

Scheme classes are loaded lazily so scheme implementation modules can import
shared scheme support without recursively importing the whole catalogue.
"""

_SCHEMES = {
    "SchemeBDF2": "stark.methods.schemes.implicit.adaptive.bdf2",
    "SchemeBackwardEuler": "stark.methods.schemes.implicit.fixed.backward_euler",
    "SchemeBogackiShampine": "stark.methods.schemes.explicit.adaptive.bogacki_shampine",
    "SchemeCashKarp": "stark.methods.schemes.explicit.adaptive.cash_karp",
    "SchemeCrankNicolson": "stark.methods.schemes.implicit.fixed.crank_nicolson",
    "SchemeCrouzeixDIRK3": "stark.methods.schemes.implicit.fixed.crouzeix_dirk3",
    "SchemeDormandPrince": "stark.methods.schemes.explicit.adaptive.dormand_prince",
    "SchemeEuler": "stark.methods.schemes.explicit.fixed.euler",
    "SchemeFehlberg45": "stark.methods.schemes.explicit.adaptive.fehlberg45",
    "SchemeGaussLegendre4": "stark.methods.schemes.implicit.fixed.gauss_legendre4",
    "SchemeHeun": "stark.methods.schemes.explicit.fixed.heun",
    "SchemeIMEXEuler": "stark.methods.schemes.imex.fixed.euler",
    "SchemeImplicitMidpoint": "stark.methods.schemes.implicit.fixed.implicit_midpoint",
    "SchemeKennedyCarpenter32": "stark.methods.schemes.imex.adaptive.kennedy_carpenter32",
    "SchemeKennedyCarpenter43_6": "stark.methods.schemes.imex.adaptive.kennedy_carpenter43_6",
    "SchemeKennedyCarpenter43_7": "stark.methods.schemes.imex.adaptive.kennedy_carpenter43_7",
    "SchemeKennedyCarpenter54": "stark.methods.schemes.imex.adaptive.kennedy_carpenter54",
    "SchemeKennedyCarpenter54b": "stark.methods.schemes.imex.adaptive.kennedy_carpenter54b",
    "SchemeKutta3": "stark.methods.schemes.explicit.fixed.kutta3",
    "SchemeKvaerno3": "stark.methods.schemes.implicit.adaptive.kvaerno3",
    "SchemeKvaerno4": "stark.methods.schemes.implicit.adaptive.kvaerno4",
    "SchemeLobattoIIIC4": "stark.methods.schemes.implicit.fixed.lobatto_iiic4",
    "SchemeMidpoint": "stark.methods.schemes.explicit.fixed.midpoint",
    "SchemeRK4": "stark.methods.schemes.explicit.fixed.rk4",
    "SchemeRK38": "stark.methods.schemes.explicit.fixed.rk38",
    "SchemeRadauIIA5": "stark.methods.schemes.implicit.fixed.radau_iia5",
    "SchemeRalston": "stark.methods.schemes.explicit.fixed.ralston",
    "SchemeSDIRK21": "stark.methods.schemes.implicit.adaptive.sdirk21",
    "SchemeSSPRK33": "stark.methods.schemes.explicit.fixed.ssprk33",
    "SchemeTsitouras5": "stark.methods.schemes.explicit.adaptive.tsitouras5",
}

__all__ = sorted(_SCHEMES)


def __getattr__(name: str):
    if name not in _SCHEMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(_SCHEMES[name]), name)
    globals()[name] = value
    return value