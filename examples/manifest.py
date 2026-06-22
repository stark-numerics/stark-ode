"""Maintained inventory of runnable STARK examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ExampleTier = Literal["getting-started", "backend", "feature", "case-study"]
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
        "examples.getting_started.returning_derivative",
        "Returning derivative",
        "getting-started",
        "Return-style derivative adapter.",
    ),
    ExampleSpec(
        "examples.getting_started.in_place_derivative",
        "In-place derivative",
        "getting-started",
        "In-place derivative adapter.",
    ),
    ExampleSpec(
        "examples.getting_started.multiple_fields",
        "Multiple fields",
        "getting-started",
        "Structured state with separate fields and translations.",
    ),
    ExampleSpec(
        "examples.getting_started.choose_scheme",
        "Choose a scheme",
        "getting-started",
        "Switch method schemes without rewriting the problem.",
    ),
    ExampleSpec(
        "examples.getting_started.checkpoints",
        "Checkpoints",
        "getting-started",
        "Ask for output checkpoints without forcing fixed solver steps.",
    ),
    ExampleSpec(
        "examples.getting_started.engine_acceleration",
        "Engine acceleration",
        "getting-started",
        "Inspect whether the NumPy engine found compiled acceleration.",
    ),
    ExampleSpec(
        "examples.backends.native",
        "Native engine",
        "backend",
        "Minimal dependency native Python engine.",
    ),
    ExampleSpec(
        "examples.backends.numpy",
        "NumPy engine",
        "backend",
        "Standard NumPy engine.",
    ),
    ExampleSpec(
        "examples.backends.jax",
        "JAX engine",
        "backend",
        "Optional JAX engine syntax.",
        cost="optional",
        default=False,
        extras=("jax",),
    ),
    ExampleSpec(
        "examples.backends.cupy",
        "CuPy engine",
        "backend",
        "Optional CuPy engine syntax.",
        cost="optional",
        default=False,
        extras=("cupy",),
    ),
    ExampleSpec(
        "examples.features.manual_stepper_setup",
        "Manual stepper setup",
        "feature",
        "Lower-level integrator setup.",
    ),
    ExampleSpec(
        "examples.features.custom_scheme_fixed_explicit",
        "Custom fixed explicit scheme",
        "feature",
        "Minimal user-defined scheme.",
    ),
    ExampleSpec(
        "examples.features.structured_state_minimal",
        "Structured state",
        "feature",
        "Nested structured state through named Frame paths.",
    ),
    ExampleSpec(
        "examples.features.foreign_model_allocator",
        "Foreign model allocator",
        "feature",
        "Custom allocator for an existing object model.",
    ),
    ExampleSpec(
        "examples.features.derivative_styles",
        "Derivative styles",
        "feature",
        "In-place, return-style, and field-kernel derivatives.",
    ),
    ExampleSpec(
        "examples.features.linearizer_styles",
        "Linearizer styles",
        "feature",
        "Linearizer styles in an implicit context.",
    ),
    ExampleSpec(
        "examples.features.norm_policy",
        "Norm policy",
        "feature",
        "Frame norm policy.",
    ),
    ExampleSpec(
        "examples.features.compare_two_schemes",
        "Compare two schemes",
        "feature",
        "Compare method choices on one problem.",
    ),
    ExampleSpec(
        "examples.features.monitor_scheme_steps",
        "Monitor scheme steps",
        "feature",
        "Observe accepted and rejected scheme steps.",
    ),
    ExampleSpec(
        "examples.features.compare_with_monitor_summary",
        "Compare with monitor summary",
        "feature",
        "Comparison reports with monitor summaries.",
    ),
    ExampleSpec(
        "examples.features.inverter_request_and_defect",
        "Inverter request and defect",
        "feature",
        "Inspect inverter request and defect records.",
    ),
    ExampleSpec(
        "examples.features.inverter_dense",
        "Dense inverter",
        "feature",
        "Direct dense correction solve.",
    ),
    ExampleSpec(
        "examples.features.inverter_krylov",
        "Krylov inverter",
        "feature",
        "Matrix-free Krylov solve and preconditioner hook.",
    ),
    ExampleSpec(
        "examples.features.scheme_predictor",
        "Scheme predictor",
        "feature",
        "Implicit stage predictor policies.",
    ),
    ExampleSpec(
        "examples.features.monitor_vs_timing",
        "Monitor versus timing",
        "feature",
        "Diagnostics versus timing measurement.",
    ),
    ExampleSpec(
        "examples.features.inverter_relaxation_richardson",
        "Richardson relaxation",
        "feature",
        "Richardson relaxation inverter.",
    ),
    ExampleSpec(
        "examples.features.inverter_relaxation_jacobi",
        "Jacobi relaxation",
        "feature",
        "Jacobi-style relaxation inverter.",
    ),
    ExampleSpec(
        "examples.features.inverter_relaxation_specialist",
        "Specialist relaxation",
        "feature",
        "Specialist relaxation inverter.",
    ),
    ExampleSpec(
        "examples.features.monitoring_levels",
        "Monitoring levels",
        "feature",
        "Different monitoring detail levels.",
    ),
    ExampleSpec(
        "examples.case_studies.three_body",
        "Three body",
        "case-study",
        "STARK as a plug-in ODE solver for an existing architecture.",
        default=False,
    ),
    ExampleSpec(
        "examples.case_studies.allen_cahn",
        "Allen-Cahn",
        "case-study",
        "PDE-like stiffness, IMEX, and implicit methods.",
        cost="slow",
        default=False,
    ),
    ExampleSpec(
        "examples.case_studies.backends",
        "Backends",
        "case-study",
        "Backend setup and timing caveats.",
        cost="optional",
        default=False,
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
