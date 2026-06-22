"""Public method-stack catalogue."""

from importlib import import_module

from stark.methods.inverters import (
    InverterConfiguration,
    InverterDense,
    InverterDescriptor,
    InverterKrylovArnoldi,
    PreconditionerDiagonalInverse,
    PreconditionerNone,
    InverterRelaxationJacobi,
    InverterRelaxationRichardson,
    InverterRelaxationSpecialist,
    InverterRelaxationStencil,
    InverterRelaxationStencilUpdate,
)
from stark.methods.method import Method, MethodError
from stark.methods.resolvents import (
    ResolventAnderson,
    ResolventBroyden,
    ResolventChord,
    ResolventConfiguration,
    ResolventCoupledNewton,
    ResolventCoupledPicard,
    ResolventDescriptor,
    ResolventError,
    ResolventNewton,
    ResolventPicard,
    ResolventVeryChord,
)
from stark.methods.schemes import __all__ as _SCHEME_NAMES

_EXPORTED_NAMES = {
    "InverterConfiguration",
    "InverterDense",
    "InverterDescriptor",
    "InverterKrylovArnoldi",
    "PreconditionerDiagonalInverse",
    "PreconditionerNone",
    "InverterRelaxationJacobi",
    "InverterRelaxationRichardson",
    "InverterRelaxationSpecialist",
    "InverterRelaxationStencil",
    "InverterRelaxationStencilUpdate",
    "Method",
    "MethodError",
    "ResolventAnderson",
    "ResolventBroyden",
    "ResolventChord",
    "ResolventConfiguration",
    "ResolventCoupledNewton",
    "ResolventCoupledPicard",
    "ResolventDescriptor",
    "ResolventError",
    "ResolventNewton",
    "ResolventPicard",
    "ResolventVeryChord",
}

__all__ = sorted(_EXPORTED_NAMES | set(_SCHEME_NAMES))


def __getattr__(name: str):
    """Lazily expose built-in schemes from `stark.methods`."""

    if name not in _SCHEME_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("stark.methods.schemes"), name)
    globals()[name] = value
    return value
