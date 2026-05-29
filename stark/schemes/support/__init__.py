"""Support objects for built-in scheme implementations."""

from stark.schemes.support.errors import unbound_scheme_call
from stark.schemes.support.descriptor import SchemeDescriptor
from stark.schemes.support.display import (
    SchemeDisplay,
    display_imex_resolvent_problem,
    display_implicit_resolvent_problem,
    with_scheme_display,
)
from stark.schemes.support.explicit import (
    SchemeSupportExplicit,
    explicit_set_apply_delta_safety,
    explicit_snapshot_state,
    initialise_explicit_support,
)
from stark.schemes.support.implicit import (
    implicit_display_resolvent_problem,
    implicit_set_apply_delta_safety,
    implicit_snapshot_state,
)
from stark.schemes.support.imex import (
    imex_display_resolvent_problem,
    imex_set_apply_delta_safety,
    imex_snapshot_state,
    initialise_imex_support,
)
from stark.schemes.support.monitoring import (
    MonitorSchemeLike,
    with_adaptive_step_monitoring,
    with_fixed_step_monitoring,
)
from stark.schemes.support.safety import SchemeSafety
from stark.schemes.support.adaptive import (
    SchemeStepAdaptiveProposal,
    SchemeStepAdaptiveAdvanceReport,
    SchemeStepControl,
    default_adaptivity,
    adaptive_adaptivity,
    initialise_adaptive_runtime,
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
    "SchemeSafety",
    "SchemeStepControl",
    "SchemeSupportExplicit",
    "MonitorSchemeLike",
    "default_adaptivity",
    "display_imex_resolvent_problem",
    "display_implicit_resolvent_problem",
    "initialise_adaptive_runtime",
    "initialise_imex_support",
    "unbound_scheme_call",
    "adaptive_adaptivity",
    "explicit_set_apply_delta_safety",
    "explicit_snapshot_state",
    "with_adaptive_step_monitoring",
    "with_fixed_step_monitoring",
    "imex_display_resolvent_problem",
    "imex_set_apply_delta_safety",
    "imex_snapshot_state",
    "implicit_display_resolvent_problem",
    "implicit_set_apply_delta_safety",
    "implicit_snapshot_state",
    "initialise_explicit_support",
    "with_scheme_display",
]
