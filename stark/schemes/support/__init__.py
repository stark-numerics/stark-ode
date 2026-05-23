"""Support objects for built-in scheme implementations."""

from stark.schemes.support.calls import unbound_scheme_call
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support.display import (
    SchemeDisplay,
    display_imex_resolvent_problem,
    display_implicit_resolvent_problem,
    with_scheme_display,
)
from stark.schemes.support.explicit import (
    SchemeSupportExplicit,
    with_explicit_workspace_methods,
    initialise_explicit_support,
)
from stark.schemes.support.implicit import with_implicit_stepper_methods
from stark.schemes.support.imex import (
    initialise_imex_support,
    with_imex_workspace_methods,
)
from stark.schemes.support.monitoring import (
    MonitorSchemeLike,
    refresh_fixed_step_call,
    with_fixed_step_monitoring,
)
from stark.schemes.support.adaptive import (
    SchemeStepAdaptiveProposal,
    SchemeStepAdaptiveAdvanceReport,
    SchemeStepControl,
    default_adaptive_regulator,
    initialise_adaptive_runtime,
    refresh_adaptive_call,
    with_adaptive_runtime_methods,
)
from stark.schemes.support.tableau import (
    ButcherTableau,
    ButcherTableauEmbedded,
    ButcherTableauImex,
)

__all__ = [
    "ButcherTableau",
    "ButcherTableauEmbedded",
    "ButcherTableauImex",
    "SchemeStepAdaptiveProposal",
    "SchemeStepAdaptiveAdvanceReport",
    "SchemeDescriptor",
    "SchemeDisplay",
    "SchemeStepControl",
    "SchemeSupportExplicit",
    "MonitorSchemeLike",
    "default_adaptive_regulator",
    "display_imex_resolvent_problem",
    "display_implicit_resolvent_problem",
    "initialise_adaptive_runtime",
    "refresh_adaptive_call",
    "refresh_fixed_step_call",
    "initialise_imex_support",
    "unbound_scheme_call",
    "with_adaptive_runtime_methods",
    "with_explicit_workspace_methods",
    "with_fixed_step_monitoring",
    "with_imex_workspace_methods",
    "with_implicit_stepper_methods",
    "initialise_explicit_support",
    "with_scheme_display",
]
