"""Maintained inventory of runnable STARK examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ExampleTier = Literal[
    "getting-started",
    "problem",
    "methods",
    "diagnostics",
    "engines",
    "inverters",
    "core",
]
ExampleCost = Literal["cheap", "optional", "slow"]


@dataclass(frozen=True, slots=True)
class ExampleSpec:
    """Describe one runnable example module.

    The manifest is the source of truth for example runners and index pages.
    Keeping the teaching purpose next to the module path makes example drift
    easier to spot during release review.
    """

    module: str
    title: str
    tier: ExampleTier
    teaches: str
    cost: ExampleCost = "cheap"
    default: bool = True
    extras: tuple[str, ...] = ()


EXAMPLES: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        "examples.getting_started.scalar_decay",
        "Scalar decay",
        "getting-started",
        "Smallest high-level solve.",
    ),
    ExampleSpec(
        "examples.getting_started.numpy_oscillator",
        "NumPy oscillator",
        "getting-started",
        "One vector-valued NumPy state field.",
    ),
    ExampleSpec(
        "examples.problem.returning_dynamics",
        "Returning dynamics",
        "problem",
        "Return-style dynamics adapter.",
    ),
    ExampleSpec(
        "examples.problem.in_place_dynamics",
        "In-place dynamics",
        "problem",
        "In-place dynamics adapter.",
    ),
    ExampleSpec(
        "examples.problem.multiple_fields",
        "Multiple fields",
        "problem",
        "Structured state with separate fields and translations.",
    ),
    ExampleSpec(
        "examples.problem.structured_state_minimal",
        "Structured state",
        "problem",
        "Nested structured state through named Frame paths.",
    ),
    ExampleSpec(
        "examples.problem.foreign_model_allocator",
        "Foreign model allocator",
        "problem",
        "Custom allocator for an existing object model.",
    ),
    ExampleSpec(
        "examples.problem.foreign_model_plug_in_solver",
        "Foreign model plug-in solver",
        "problem",
        "Replace an existing time stepper without flattening the user's object model.",
    ),
    ExampleSpec(
        "examples.problem.foreign_model_audit",
        "Foreign model audit",
        "problem",
        "Audit a custom adapter for an existing object model.",
    ),
    ExampleSpec(
        "examples.problem.reaction_diffusion_array",
        "Reaction-diffusion array",
        "problem",
        "PDE-like array state through the high-level System path.",
    ),
    ExampleSpec(
        "examples.problem.dynamics_styles",
        "Dynamics styles",
        "problem",
        "In-place, return-style, and field-kernel dynamics.",
    ),
    ExampleSpec(
        "examples.problem.linearizer_styles",
        "Linearizer styles",
        "problem",
        "Linearizer styles in an implicit context.",
    ),
    ExampleSpec(
        "examples.problem.norm_policy",
        "Norm policy",
        "problem",
        "Frame norm policy.",
    ),
    ExampleSpec(
        "examples.methods.choose_scheme",
        "Choose a scheme",
        "methods",
        "Switch method schemes without rewriting the problem.",
    ),
    ExampleSpec(
        "examples.methods.custom_scheme_fixed_explicit",
        "Custom fixed explicit scheme",
        "methods",
        "Minimal user-defined scheme.",
    ),
    ExampleSpec(
        "examples.methods.imex_with_custom_spectral_resolvent",
        "IMEX custom spectral resolvent",
        "methods",
        "Exploit a diagonal Fourier-space operator inside an IMEX stage solve.",
    ),
    ExampleSpec(
        "examples.methods.matrix_free_jacobian",
        "Matrix-free Jacobian",
        "methods",
        "Newton with a Jacobian action and Krylov linear correction solve.",
    ),
    ExampleSpec(
        "examples.methods.resolvent_fixed_point",
        "Fixed-point resolvent",
        "methods",
        "Picard iteration for a mild implicit stage.",
    ),
    ExampleSpec(
        "examples.methods.resolvent_linearized",
        "Linearized resolvent",
        "methods",
        "Newton iteration with a linearizer and dense inverter.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_explicit_fixed",
        "Explicit fixed scheme",
        "methods",
        "Fixed-step explicit time integration.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_explicit_adaptive",
        "Explicit adaptive scheme",
        "methods",
        "Error-controlled explicit time integration.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_implicit_fixed",
        "Implicit fixed scheme",
        "methods",
        "Fixed-step implicit integration with a resolvent.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_implicit_adaptive",
        "Implicit adaptive scheme",
        "methods",
        "Adaptive implicit integration with a resolvent.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_imex_fixed",
        "IMEX fixed scheme",
        "methods",
        "Fixed-step implicit-explicit split integration.",
    ),
    ExampleSpec(
        "examples.methods.scheme_family_imex_adaptive",
        "IMEX adaptive scheme",
        "methods",
        "Adaptive implicit-explicit split integration.",
    ),
    ExampleSpec(
        "examples.methods.scheme_predictor",
        "Scheme predictor",
        "methods",
        "Implicit stage predictor policies.",
    ),
    ExampleSpec(
        "examples.diagnostics.checkpoints",
        "Checkpoints",
        "diagnostics",
        "Ask for output checkpoints without forcing fixed solver steps.",
    ),
    ExampleSpec(
        "examples.diagnostics.compare_two_schemes",
        "Compare two schemes",
        "diagnostics",
        "Compare method choices on one problem.",
    ),
    ExampleSpec(
        "examples.diagnostics.compare_custom_scheme",
        "Compare custom scheme",
        "diagnostics",
        "Compare a user-defined scheme through the Method API.",
    ),
    ExampleSpec(
        "examples.diagnostics.monitor_scheme_steps",
        "Monitor scheme steps",
        "diagnostics",
        "Observe accepted and rejected scheme steps.",
    ),
    ExampleSpec(
        "examples.diagnostics.error_ratio_trace",
        "Error-ratio trace",
        "diagnostics",
        "Interpret adaptive scheme error ratios over time.",
    ),
    ExampleSpec(
        "examples.diagnostics.compare_with_monitor_summary",
        "Compare with monitor summary",
        "diagnostics",
        "Comparison reports with monitor summaries.",
    ),
    ExampleSpec(
        "examples.diagnostics.monitor_vs_timing",
        "Monitor versus timing",
        "diagnostics",
        "Diagnostics versus timing measurement.",
    ),
    ExampleSpec(
        "examples.diagnostics.monitoring_levels",
        "Monitoring levels",
        "diagnostics",
        "Different monitoring detail levels.",
    ),
    ExampleSpec(
        "examples.engines.engine_acceleration",
        "Engine acceleration",
        "engines",
        "Inspect whether the NumPy engine found compiled acceleration.",
    ),
    ExampleSpec(
        "examples.engines.backend_native",
        "Native engine",
        "engines",
        "Minimal dependency native Python engine.",
    ),
    ExampleSpec(
        "examples.engines.backend_numpy",
        "NumPy engine",
        "engines",
        "Standard NumPy engine.",
    ),
    ExampleSpec(
        "examples.engines.backend_jax",
        "JAX engine",
        "engines",
        "Optional JAX engine syntax.",
        cost="optional",
        default=False,
        extras=("jax",),
    ),
    ExampleSpec(
        "examples.engines.backend_cupy",
        "CuPy engine",
        "engines",
        "Optional CuPy engine syntax.",
        cost="optional",
        default=False,
        extras=("cupy",),
    ),
    ExampleSpec(
        "examples.inverters.inverter_request_and_defect",
        "Inverter request and defect",
        "inverters",
        "Inspect inverter request and defect records.",
    ),
    ExampleSpec(
        "examples.inverters.inverter_dense",
        "Dense inverter",
        "inverters",
        "Direct dense correction solve.",
    ),
    ExampleSpec(
        "examples.inverters.inverter_krylov",
        "Krylov inverter",
        "inverters",
        "Matrix-free Krylov solve and preconditioner hook.",
    ),
    ExampleSpec(
        "examples.inverters.inverter_relaxation_richardson",
        "Richardson relaxation",
        "inverters",
        "Richardson relaxation inverter.",
    ),
    ExampleSpec(
        "examples.inverters.inverter_relaxation_jacobi",
        "Jacobi relaxation",
        "inverters",
        "Jacobi-style relaxation inverter.",
    ),
    ExampleSpec(
        "examples.inverters.inverter_relaxation_linear_fixed",
        "LinearFixed relaxation",
        "inverters",
        "LinearFixed relaxation inverter.",
    ),
    ExampleSpec(
        "examples.core.manual_stepper_setup",
        "Manual stepper setup",
        "core",
        "Lower-level integrator setup.",
    ),
)


def examples_for_tier(
    tier: ExampleTier,
    *,
    include_optional: bool = False,
    include_slow: bool = False,
) -> tuple[ExampleSpec, ...]:
    """Return default examples for a teaching tier."""

    return tuple(
        example
        for example in EXAMPLES
        if example.tier == tier
        and example.default
        and (include_optional or example.cost != "optional")
        and (include_slow or example.cost != "slow")
    )
